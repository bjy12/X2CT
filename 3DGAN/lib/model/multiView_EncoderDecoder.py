# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import matplotlib.pyplot as plt
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory
from .loss.multi_gan_loss import RestructionLoss
import lib.utils.metrics as Metrics
import numpy as np
import pdb

class CTGAN(Base_Model):
  def __init__(self):
    super(CTGAN, self).__init__()

  @property
  def name(self):
    return 'multiView_ED3D'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.if_pool = opt.if_pool
    self.multi_view = opt.multi_view

    assert len(self.multi_view) > 0

    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    #self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']
    self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2']

    #pdb.set_trace()
    self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                   opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                   opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                   opt.init_type, self.gpu_ids, opt.n_blocks,
                                   opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                   opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt)

    self.loss_names = ['idt']
    self.model_names = ['G']

    # map loss
    if self.opt.map_projection_lambda > 0:
      self.loss_names += ['map_m']


  # correspond to visual_names
  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
      [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
    ]

  def init_loss(self, opt):
    Base_Model.init_loss(self, opt)

    # #####################
    # define loss functions
    # #####################

    # identity loss
    self.criterionIdt = RestructionLoss(opt.idt_loss, opt.idt_reduction).to(self.device)

    # map loss
    self.criterionMap = RestructionLoss(opt.map_projection_loss).to(self.device)

    # #####################
    # initialize optimizers
    # #####################
    self.optimizers = []
    if self.opt.weight_decay_if:
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                          lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=1e-4)
    else:
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                          lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizers.append(self.optimizer_G)

  '''
    Train -Forward and Backward
  '''
  def check_order(self, input):
    xray1_data = input[1][0][0]
    xray2_data = input[2][0][0]
    ct_data = input[0][0]
    pdb.set_trace()

    # 创建图像网格
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Input Data Visualization', fontsize=16)
    
    # 显示CT数据的三个正交面
    depth = ct_data.shape[0]
    height = ct_data.shape[1]
    width = ct_data.shape[2]
    
    # 选择中间切片
    axes[0,0].imshow(ct_data[depth//2, :, :].numpy(), cmap='gray')
    axes[0,0].set_title('CT - Axial (Middle Slice)')
    
    axes[0,1].imshow(ct_data[:, height//2, :].numpy(), cmap='gray')
    axes[0,1].set_title('CT - Sagittal (Middle Slice)')
    
    axes[0,2].imshow(ct_data[:, :, width//2].numpy(), cmap='gray')
    axes[0,2].set_title('CT - Coronal (Middle Slice)')
    
    # 显示X-ray图像
    pdb.set_trace()
    axes[1,0].imshow(xray1_data[0].numpy(), cmap='gray')
    axes[1,0].set_title('X-ray 1')
    
    axes[1,1].imshow(xray2_data[0].numpy(), cmap='gray')
    axes[1,1].set_title('X-ray 2')
    pdb.set_trace()
    # 显示数据信息
    info_text = f"""Data Information:
    CT shape: {ct_data.shape}
    X-ray1 shape: {xray1_data.shape}
    X-ray2 shape: {xray2_data.shape}"""
    axes[1,2].text(0.1, 0.5, info_text, fontsize=10)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
  def check_orderv2(self, input):
    xray1_data = input[1][0][0]
    xray2_data = input[2][0][0]
    ct_data = input[0][0]
    pdb.set_trace()
    
    xray1_ct = xray1_data[0].unsqueeze(-1)
    xray1_ct = xray1_ct.repeat(1,1,xray1_ct.shape[0])
    xray2_ct = xray2_data[0].unsqueeze(-1)
    xray2_ct = xray2_ct.repeat(1,1,xray2_ct.shape[0])

    # 创建图像网格
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Input Data Visualization', fontsize=16)
    
    # 显示CT数据的三个正交面
    depth = ct_data.shape[0]
    height = ct_data.shape[1]
    width = ct_data.shape[2]
    
    # 选择中间切片
    axes[0,0].imshow(ct_data[depth//2, :, :].numpy(), cmap='gray')
    axes[0,0].set_title('CT - Axial (Middle Slice)')
    
    axes[0,1].imshow(ct_data[:, height//2, :].numpy(), cmap='gray')
    axes[0,1].set_title('CT - Sagittal (Middle Slice)')
    
    axes[0,2].imshow(ct_data[:, :, width//2].numpy(), cmap='gray')
    axes[0,2].set_title('CT - Coronal (Middle Slice)')
    
    # 显示X-ray图像
    axes[1,0].imshow(xray1_ct[depth//2, :, :].numpy(), cmap='gray')
    axes[1,0].set_title('xary_1 - Axial (Middle Slice)')
    
    axes[1,1].imshow(xray1_ct[:, height//2, :].numpy(), cmap='gray')
    axes[1,1].set_title('xary_1 - Sagittal (Middle Slice)')
    
    axes[1,2].imshow(xray1_ct[:, :, width//2].numpy(), cmap='gray')
    axes[1,2].set_title('xary_1 - Coronal (Middle Slice)')
    
    axes[2,0].imshow(xray2_ct[depth//2, :, :].numpy(), cmap='gray')
    axes[2,0].set_title('xary_2 - Axial (Middle Slice)')
    
    axes[2,1].imshow(xray2_ct[:, height//2, :].numpy(), cmap='gray')
    axes[2,1].set_title('xary_2 - Sagittal (Middle Slice)')
    
    axes[2,2].imshow(xray2_ct[:, :, width//2].numpy(), cmap='gray')
    axes[2,2].set_title('xary_2 - Coronal (Middle Slice)')
    pdb.set_trace()
    # 显示数据信息
    info_text = f"""Data Information:
    CT shape: {ct_data.shape}
    X-ray1 shape: {xray1_data.shape}
    X-ray2 shape: {xray2_data.shape}"""
    axes[1,2].text(0.1, 0.5, info_text, fontsize=10)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
  def preprocess_data(self , ct_data, xray1, xray2):
      """预处理数据以确保方向正确"""
      # 1. 首先确保CT数据方向正确
      #pdb.set_trace()
      #ct_data = ct_data.permute(0, 2, 1)  # 调整CT的轴向排列
      pdb.set_trace()
      # 2. 确保X-ray方向与CT投影匹配
      #xray1 = xray1.transpose(1, 0)  # 调整正面X-ray方向
      xray1 = torch.flip(xray1, [0])  # 如果需要翻转侧面X-ray
      xray2 = xray2.transpose(1,0).flip([1])
      return ct_data, xray1, xray2
  
  def preprocess_data2(self , data):
      xray1 = data[1][0].to(self.device)
      xray2 = data[2][0].to(self.device)
      ct = data[0]
      
      xray1 = torch.flip(xray1 , [2])
      xray2 = xray2.permute(0,1,3,2).flip([3])

      
      #pdb.set_trace()

      return  xray1 , xray2  , ct 
      
  def enhanced_check_order(self, input):
      """
      Enhanced function to check data order and visualize results
      """
      xray1_data = input[1][0][0][0]  # [H, W]
      xray2_data = input[2][0][0][0]  # [H, W]
      ct_data = input[0][0]        # [D, H, W]
      
      pdb.set_trace()
      ct_data , xray1_data , xray2_data= self.preprocess_data(ct_data, xray1_data, xray2_data)

      # 创建更大的图像网格用于详细显示
      fig = plt.figure(figsize=(20, 15))
      gs = plt.GridSpec(3, 4)
      
      # 1. CT 数据可视化 (三个正交面)
      depth, height, width = ct_data.shape
      
      # Axial view (D方向)
      ax1 = fig.add_subplot(gs[0, 0])
      ax1.imshow(ct_data[depth//2, :, :].numpy(), cmap='gray')
      ax1.set_title(f'CT Axial (D={depth//2})\nShape: {ct_data.shape}')
      ax1.axis('on')
      
      # Sagittal view (H方向)
      ax2 = fig.add_subplot(gs[0, 1])
      ax2.imshow(ct_data[:, height//2, :].numpy(), cmap='gray')
      ax2.set_title(f'CT Sagittal (H={height//2})')
      ax2.axis('on')
      
      # Coronal view (W方向)
      ax3 = fig.add_subplot(gs[0, 2])
      ax3.imshow(ct_data[:, :, width//2].numpy(), cmap='gray')
      ax3.set_title(f'CT Coronal (W={width//2})')
      ax3.axis('on')
      
      # 2. X-ray数据可视化
      # X-ray1
      ax4 = fig.add_subplot(gs[1, 0])
      ax4.imshow(xray1_data.numpy(), cmap='gray')
      ax4.set_title(f'X-ray1 Original\nShape: {xray1_data.shape}')
      ax4.axis('on')
      
      # X-ray2
      ax5 = fig.add_subplot(gs[1, 1])
      ax5.imshow(xray2_data.numpy(), cmap='gray')
      ax5.set_title(f'X-ray2 Original\nShape: {xray2_data.shape}')
      ax5.axis('on')
      
      # 3. CT投影模拟
      # 沿D方向的投影（模拟正面X-ray）
      proj_d = torch.mean(ct_data, dim=0).numpy()
      ax6 = fig.add_subplot(gs[2, 0])
      ax6.imshow(proj_d, cmap='gray')
      ax6.set_title('CT Projection (along D)\nShould match X-ray1')
      ax6.axis('on')
      
      # 沿W方向的投影（模拟侧面X-ray）
      proj_w = torch.mean(ct_data, dim=2).numpy()
      ax7 = fig.add_subplot(gs[2, 1])
      ax7.imshow(proj_w, cmap='gray')
      ax7.set_title('CT Projection (along W)\nShould match X-ray2')
      ax7.axis('on')
      
      # 4. 添加数值信息
      info_text = f"""Data Statistics:
      CT:
        Shape: {ct_data.shape}
        Value Range: [{ct_data.min():.2f}, {ct_data.max():.2f}]
        Mean: {ct_data.mean():.2f}
      X-ray1:
        Shape: {xray1_data.shape}
        Value Range: [{xray1_data.min():.2f}, {xray1_data.max():.2f}]
        Mean: {xray1_data.mean():.2f}
      X-ray2:
        Shape: {xray2_data.shape}
        Value Range: [{xray2_data.min():.2f}, {xray2_data.max():.2f}]
        Mean: {xray2_data.mean():.2f}
      """
      
      ax_text = fig.add_subplot(gs[:, 3])
      ax_text.text(0.1, 0.5, info_text, fontsize=12, va='center')
      ax_text.axis('off')
      
      plt.tight_layout()
      plt.show()
      
      # 5. 检查数据的对应关系
      print("\nChecking View Correspondence:")
      print("-" * 50)
      
      # X-ray1 与 CT投影的相似度
      similarity_xray1 = torch.nn.functional.mse_loss(
          torch.mean(ct_data, dim=0),
          xray1_data
      )
      print(f"X-ray1 to CT projection similarity (MSE): {similarity_xray1.item():.4f}")
      
      # X-ray2 与 CT投影的相似度
      similarity_xray2 = torch.nn.functional.mse_loss(
          torch.mean(ct_data, dim=2),
          xray2_data
      )
      print(f"X-ray2 to CT projection similarity (MSE): {similarity_xray2.item():.4f}")
  def set_input(self, input):
    #pdb.set_trace()
    #self.check_order(input)
    #self.check_orderv2(input)
    #self.enhanced_check_order(input)
    #pdb.set_trace()
    xray1 , xray2 , ct = self.preprocess_data2(input)
    self.G_input1 = xray1.to(self.device)
    self.G_input2 = xray2.to(self.device)
    self.G_real = ct.to(self.device)
    #self.image_paths = input[2:]

  # map function
  def output_map(self, v, dim):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(v), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.mean(torch.abs(v), dim=dim)
      return map
    else:
      raise NotImplementedError()

  def transition(self, predict):
    p_max, p_min = predict.max(), predict.min()
    new_predict = (predict - p_min) / (p_max - p_min)
    return new_predict

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def ct_Gaussian(self, value):
    return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

  def post_process(self, attributes_name):
    if not self.training:
      if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
      elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
      else:
        raise NotImplementedError()

  def projection_visual(self):
    # map F is projected in dimension of H
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    # map S is projected in dimension of W
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

  def metrics_evaluation(self):
    # 3D metrics including mse, cs and psnr
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)

    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  def forward(self):
    '''
    self.G_fake is generated object
    self.G_real is GT object
    '''
    """
    input xray1 xray2 
    """
    # G_fake_D is [B 1 D H W]
    #pdb.set_trace()
    self.G_fake_D1, self.G_fake_D2, self.G_fake_D = self.netG([self.G_input1, self.G_input2])
    # visual object should be [B D H W]
    #pdb.set_trace()
    self.G_fake = torch.squeeze(self.G_fake_D, 1)
    # input of Discriminator is [B 1 D H W]
    #pdb.set_trace()
    self.G_real_D = torch.unsqueeze(self.G_real, 1)
    self.G_real_D1 = self.G_real_D.clone().permute(*self.opt.CTOrder_Xray1)
    self.G_real_D2 = self.G_real_D.clone().permute(*self.opt.CTOrder_Xray2)
    # post processing, used only in testing
    self.post_process(['G_fake'])
    if not self.training:
      # visualization of x-ray projection
      self.projection_visual()
      # metrics
      self.metrics_evaluation()
    # multi-view projection maps for training
    # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
    if self.training:
      for i in self.multi_view:
        #pdb.set_trace()
        out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_real'.format(i), out_map)

        out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_fake'.format(i), out_map)

  def optimize_parameters(self):
    # forward 
    self()
    self.optimizer_G.zero_grad()

    total_loss = 0
    idt_lambda = self.opt.idt_lambda
    map_projection_lambda = self.opt.map_projection_lambda

    # focus area weight assignment
    if self.opt.idt_reduction == 'none' and self.opt.idt_weight > 0:
      idt_low, idt_high = self.opt.idt_weight_range
      idt_weight = self.opt.idt_weight
      loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
      mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
      loss_idt[mask] = loss_idt[mask] * idt_weight
      self.loss_idt = loss_idt.mean() * idt_lambda
    else:
      self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda
    total_loss += self.loss_idt

    if self.opt.map_projection_lambda > 0:
      self.loss_map_m = 0.
      for direction in self.multi_view:
        self.loss_map_m += self.criterionMap(
          getattr(self, 'G_Map_{}_fake'.format(direction)),
          getattr(self, 'G_Map_{}_real'.format(direction))) * map_projection_lambda
      self.loss_map_m = self.loss_map_m / len(self.multi_view)
      total_loss += self.loss_map_m
    total_loss.backward()
    self.optimizer_G.step()
