from torch import floor, remainder
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
import time
import math


def main(args):
	''' Create Dir for Save'''
	log_dir, checkpoints_dir, val_dir = create_dir(args)

	''' Logger '''
	logger = Logger(log_dir, args)

	''' CPU or Cuda'''
	device = torch.device(args.device)
	if 'cuda' in args.device:
		torch.cuda.set_device(device)

	''' DATA Training LOADING '''
	logger.log_string('\nLoad Training Dataset ...')
	train_Dataset = TrainSetDataLoader(args)
	logger.log_string("The number of training data is: %d" % len(train_Dataset))
	train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
											   batch_size=args.batch_size, shuffle=True,)

	''' DATA Validation LOADING '''
	logger.log_string('\nLoad Validation Dataset ...')
	test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
	logger.log_string("The number of validation data is: %d" % length_of_tests)


	''' MODEL LOADING '''
	logger.log_string('\nModel Initial ...')
	MODEL_PATH = 'model.' + args.task + '.' + args.model_name
	MODEL = importlib.import_module(MODEL_PATH)
	net = MODEL.get_model(args)


	''' Load Pre-Trained PTH '''
	if args.use_pre_ckpt == False:
		net.apply(MODEL.weights_init)
		start_epoch = 0
		logger.log_string('Do not use pre-trained model!')
	else:
		try:
			ckpt_path = args.path_pre_pth
			checkpoint = torch.load(ckpt_path, map_location='cpu')
			start_epoch = checkpoint['epoch']
			try:
				new_state_dict = OrderedDict()
				for k, v in checkpoint['state_dict'].items():
					name = 'module.' + k  # add `module.`
					new_state_dict[name] = v
				# load params
				net.load_state_dict(new_state_dict)
				logger.log_string('Use pretrain model!')
			except:
				new_state_dict = OrderedDict()
				for k, v in checkpoint['state_dict'].items():
					new_state_dict[k] = v
				# load params
				net.load_state_dict(new_state_dict)
				logger.log_string('Use pretrain model!')
		except:
			net = MODEL.get_model(args)
			net.apply(MODEL.weights_init)
			start_epoch = 0
			logger.log_string('No existing model, starting training from scratch...')
			pass
		pass
	
	net = net.to(device)
	cudnn.benchmark = True

	if args.parallel:
		net = torch.nn.DataParallel(net, device_ids=[0, 1])

	total_params = sum(p.numel() for p in net.parameters())
	print("\nTotal Params: {:.2f}\n".format(total_params)) 

	''' Print Parameters '''
	logger.log_string('PARAMETER ...')
	logger.log_string(args)


	''' LOSS LOADING '''
	criterion = MODEL.get_loss(args).to(device)


	''' Optimizer '''
	optimizer = torch.optim.Adam(
		[paras for paras in net.parameters() if paras.requires_grad == True],
		lr=args.lr,
		betas=(0.9, 0.999),
		eps=1e-08,
		weight_decay=args.decay_rate
	)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


	''' TRAINING & TEST '''
	logger.log_string('\nStart training...')
	for idx_epoch in range(start_epoch, args.epoch):
		logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

		''' Training '''
		loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer)
		logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
						  (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))


		''' Save PTH  '''
		if args.local_rank == 0:
			save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
			args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
			state = {
				'epoch': idx_epoch + 1,
				'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
			}
			torch.save(state, save_ckpt_path)
			logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


		''' Validation '''
		step = args.step_validation
		if (idx_epoch + 1)%step==0 or idx_epoch > args.epoch-step:
			with torch.no_grad():
				''' Create Excel for PSNR/SSIM '''
				excel_file = ExcelFile()

				psnr_testset = []
				ssim_testset = []
				for index, test_name in enumerate(test_Names):
					test_loader = test_Loaders[index]

					epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
					epoch_dir.mkdir(exist_ok=True)
					save_dir = epoch_dir.joinpath(test_name)
					save_dir.mkdir(exist_ok=True)

					psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
					

					excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)


					psnr_epoch_test = float(np.array(psnr_iter_test).mean())
					ssim_epoch_test = float(np.array(ssim_iter_test).mean())


					psnr_testset.append(psnr_epoch_test)
					ssim_testset.append(ssim_epoch_test)
					logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
					idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
					pass
				psnr_mean_test = float(np.array(psnr_testset).mean())
				ssim_mean_test = float(np.array(ssim_testset).mean())
				logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
								  % (psnr_mean_test, ssim_mean_test))
				excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
				pass
			pass

		''' scheduler '''
		scheduler.step()
		pass
	pass


def train(train_loader, device, net, criterion, optimizer):
	''' training one epoch '''
	psnr_iter_train = []
	loss_iter_train = []
	ssim_iter_train = []
	for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
		[Lr_angRes_in, Lr_angRes_out] = data_info
		data_info[0] = Lr_angRes_in[0].item()
		data_info[1] = Lr_angRes_out[0].item()

		data = data.to(device)      # low resolution
		label = label.to(device)    # high resolution
		out = net(data, data_info)
		loss = criterion(out, label, data_info)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		torch.cuda.empty_cache()

		loss_iter_train.append(loss.data.cpu())
		psnr, ssim = cal_metrics(args, label, out)
		psnr_iter_train.append(psnr)
		ssim_iter_train.append(ssim)
		pass

	loss_epoch_train = float(np.array(loss_iter_train).mean())
	psnr_epoch_train = float(np.array(psnr_iter_train).mean())
	ssim_epoch_train = float(np.array(ssim_iter_train).mean())

	return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


def test(test_loader, device, net, save_dir=None):
	LF_iter_test = []
	psnr_iter_test = []
	ssim_iter_test = []
	

	for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
		time_total  = 0
		[Lr_angRes_in, Lr_angRes_out] = data_info
		data_info[0] = Lr_angRes_in[0].item()
		data_info[1] = Lr_angRes_out[0].item()

		def forward_chop(Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr,net , crop_test_method=3):
			time_total = 0

			if args.crop_test_method == 1:
			
				Lr_SAI_y = Lr_SAI_y.to(device)  # numU, numV, h*angRes, w*angRes
				Lr_SAI_y = SAI2MacPI(Lr_SAI_y,args.angRes_out)

				Hr_SAI_y = Hr_SAI_y
				Sr_SAI_cbcr = Sr_SAI_cbcr

				b, c, h, w = Lr_SAI_y.size()

				H = h/args.angRes_out
				W = w/args.angRes_out

				bdr_h = 0
				bdr_w = 0

				if (H % 4) !=0:
					bdr_h = ((math.ceil(H / 4) * 4)*args.angRes_out - h)
					Lr_SAI_y = ImageExtend(Lr_SAI_y, [bdr_h, 0, 0, 0])

				if (W % 4) !=0:
					bdr_w = ((math.ceil(W / 4) * 4)*args.angRes_out - w)
					Lr_SAI_y = ImageExtend(Lr_SAI_y, [0, 0, bdr_w, 0])

				

				with torch.no_grad():
					Lr_SAI_y = MacPI2SAI(Lr_SAI_y,args.angRes_out)
					time_item_start = time.time()
					Sr_SAI_y = net(Lr_SAI_y)
					time_total += time.time() - time_item_start
				
				Sr_SAI_y = SAI2MacPI(Sr_SAI_y,args.angRes_out)
				b, c, h, w = Sr_SAI_y.size()

				if (H % 4) !=0:
					bdr_h = args.scale_factor*bdr_h
					Sr_SAI_y = Sr_SAI_y[:,:,bdr_h:h, :]

				if (W % 4) !=0:
					bdr_w = args.scale_factor*bdr_w
					Sr_SAI_y = Sr_SAI_y[:,:,:, bdr_w:w]

				Sr_SAI_y = MacPI2SAI(Sr_SAI_y,args.angRes_out)

			

			if args.crop_test_method == 2:
			
				Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
				Hr_SAI_y = Hr_SAI_y
				Sr_SAI_cbcr = Sr_SAI_cbcr

				''' Crop LFs into Patches '''
				subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
				numU, numV, H, W = subLFin.size()
				subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
				subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
									args.angRes_in * args.patch_size_for_test * args.scale_factor)

				''' SR the Patches '''
				for i in range(0, numU * numV, args.minibatch_for_test):
					tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
					with torch.no_grad():
						time_item_start = time.time()
						net.eval()
						torch.cuda.empty_cache()
						out = net(tmp.to(device), data_info)
						time_total += time.time() - time_item_start
						subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
				subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

				''' Restore the Patches to LFs '''
				Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
									args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
				Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
				Sr_SAI_y = Sr_SAI_y.to(args.device)

			if args.crop_test_method == 3:

				Lr_SAI_y = Lr_SAI_y.to(args.device)  # numU, numV, h*angRes, w*angRes
				Lr_SAI_y = SAI2MacPI(Lr_SAI_y,args.angRes_out)
			
				b, c, h, w = Lr_SAI_y.size()

				H = h/args.angRes_out
				W = w/args.angRes_out

				bdr_h = 0
				bdr_w = 0

				if (H % 4) !=0:
					bdr_h = ((math.ceil(H / 4) * 4)*args.angRes_out - h)
					Lr_SAI_y = ImageExtend(Lr_SAI_y, [bdr_h, 0, 0, 0])

				if (W % 4) !=0:
					bdr_w = ((math.ceil(W / 4) * 4)*args.angRes_out - w)
					Lr_SAI_y = ImageExtend(Lr_SAI_y, [0, 0, bdr_w, 0])

				
				b, c, h, w = Lr_SAI_y.size()

				scale  = args.scale_factor
				h_half, w_half = h // 2, w // 2
				h_size, w_size = h_half , w_half 

				lr_list = [
					Lr_SAI_y[:, :, 0:h_size, 0:w_size],
					Lr_SAI_y[:, :, 0:h_size, (w - w_size):w],
					Lr_SAI_y[:, :, (h - h_size):h, 0:w_size],
					Lr_SAI_y[:, :, (h - h_size):h, (w - w_size):w]]


				sr_list = []
				n_GPUs = 1
				for i in range(0, 4, n_GPUs):
					lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
					lr_batch = MacPI2SAI(lr_batch, args.angRes_out)
					with torch.no_grad():
						time_item_start = time.time()
						sr_batch = net(lr_batch)
						time_total += time.time() - time_item_start
					sr_batch = SAI2MacPI(sr_batch,args.angRes_out)
					sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))

				h, w = scale * h, scale * w
				h_half, w_half = scale * h_half, scale * w_half
				h_size, w_size = scale * h_size, scale * w_size
			
				Sr_SAI_y = Lr_SAI_y.new(b, c, h, w)
				Sr_SAI_y[:, :, 0:h_half, 0:w_half] \
					= sr_list[0][:, :, 0:h_half, 0:w_half]
				Sr_SAI_y[:, :, 0:h_half, w_half:w] \
					= sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
				Sr_SAI_y[:, :, h_half:h, 0:w_half] \
					= sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
				Sr_SAI_y[:, :, h_half:h, w_half:w] \
					= sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

				if (H % 4) !=0:
					bdr_h = args.scale_factor*bdr_h
					Sr_SAI_y = Sr_SAI_y[:,:,bdr_h:h, :]

				if (W % 4) !=0:
					bdr_w = args.scale_factor*bdr_w
					Sr_SAI_y = Sr_SAI_y[:,:,:, bdr_w:w]
				
				Sr_SAI_y = MacPI2SAI(Sr_SAI_y,args.angRes_out)
				
			return Sr_SAI_y

		if args.self_ensemble:

			def _transform(v, op):
				if args.precision != 'single': v = v.float()

				v2np = v.data.cpu().numpy()
				if op == 'v':
					tfnp = v2np[:, :, :, ::-1].copy()
				elif op == 'h':
					tfnp = v2np[:, :, ::-1, :].copy()
				elif op == 't':
					tfnp = v2np.transpose((0, 1, 3, 2)).copy()

				ret = torch.Tensor(tfnp).to(args.device)
				if args.precision == 'half': ret = ret.half()

				return ret

			lr_list = [Lr_SAI_y]
			for tf in 'v', 'h', 't':
				lr_list.extend([_transform(t, tf) for t in lr_list])

			sr_list = [forward_chop(Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, net , crop_test_method=args.crop_test_method) for Lr_SAI_y in lr_list]
			for i in range(len(sr_list)):
				if i > 3:
					sr_list[i] = _transform(sr_list[i], 't')
				if i % 4 > 1:
					sr_list[i] = _transform(sr_list[i], 'h')
				if (i % 4) % 2 == 1:
					sr_list[i] = _transform(sr_list[i], 'v')

			output_cat = torch.cat(sr_list, dim=0)
			Sr_SAI_y = output_cat.mean(dim=0, keepdim=True)

		else:
			Sr_SAI_y = forward_chop(Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr,net , crop_test_method=args.crop_test_method)

			
		''' Calculate the PSNR & SSIM '''
		if args.test_NTIRE2023_LFSR == 1:
			# just keep here for consistency
			psnr_iter_test = 0
			ssim_iter_test = 0
		else:
			psnr, ssim = cal_metrics_2(Hr_SAI_y.squeeze(), Sr_SAI_y.squeeze(),args.angRes_out )
			psnr_iter_test.append(psnr)
			ssim_iter_test.append(ssim)
			LF_iter_test.append(LF_name[0])
			
		
		''' Save RGB '''
		if save_dir is not None:
			save_dir_ = save_dir.joinpath(LF_name[0])
			save_dir_.mkdir(exist_ok=True)
			# views_dir = save_dir_.joinpath('views')
			views_dir = save_dir_
			views_dir.mkdir(exist_ok=True)
			Sr_SAI_y = Sr_SAI_y.to('cpu')
			Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
			Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
			Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

			# save the SAI
			# path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
			# imageio.imwrite(path, Sr_SAI_rgb)
		
			# save the center view
			# img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
			# path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
			# imageio.imwrite(path, img)
			# save all views
			for i in range(args.angRes_out):
				for j in range(args.angRes_out):
					img = Sr_4D_rgb[i, j, :, :, :]
					path = str(views_dir) + '/View'  + '_' + str(i) + '_' + str(j) + '.bmp'
					imageio.imwrite(path, img)
					pass
				pass
			pass
		pass

	return psnr_iter_test, ssim_iter_test, LF_iter_test


if __name__ == '__main__':
	from option import args

	main(args)
