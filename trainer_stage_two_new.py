from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import numpy as np
import torch.optim as optim
import torch.nn as nn

from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # 字典
        self.parameters_to_train = []  # 列表

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")  # 18
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["position_encoder"].to(self.device)

        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales)

        self.models["position"].to(self.device)

        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train_real"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'  

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)


        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        self.models["encoder"].train()
        self.models["depth"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["encoder"].eval()
        self.models["depth"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        #Not used
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            
            #DepthNet Prediction
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
        
        #Not used
        #if self.opt.predictive_mask:
        #    outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features, disps):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #print(pose_feats[0].shape)
            #print(pose_feats[-1].shape)
            #print(pose_feats[1].shape)
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    #print("inputs_all",inputs_all[0].shape)
                    #print("inputs_all_reverse",inputs_all[1].shape)
                    
                    # OF Prediction
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)
                    flow_2D = position_inputs
                     # recover image shape, to avoid meaningless flow matches
                    if h_side is not None or w_side is not None:
                        flow_2D = flow_2D[:,:,:h_side,:w_side]
                    try:  
                        conf = conf[:,:,:h_side,:w_side]   
                    except:
                        conf = conf

                    # some inputs are left for possible visualization or debug, plz ignore them if not
                    # return:   Pose matrix             Bx3x4
                    #           Essential matrix        Bx3x3
                    P_mat,E_mat = self.pose_by_ransac(flow_2D,ref,target,intrinsic_inv_gpu,
                                                        h_side,w_side,pose_gt =pose_gt,img_path=img_path)
                    #print(outputs_0['position', 0][0][0])
                    #print(outputs_0['position', 0][0][1])
                    #print(outputs_1['position', 0][0][1].shape)
                    #print(len(outputs_1))
                    for scale in self.opt.scales:
                        outputs["p_"+str(scale)+"_"+str(f_i)] = outputs_0["position_"+str(scale)]
                        outputs["ph_"+str(scale)+"_"+str(f_i)] = F.interpolate(
                            outputs["p_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        outputs["r_"+str(scale)+"_"+str(f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs["ph_"+str(scale)+"_"+str(f_i)])
                        outputs["pr_"+str(scale)+"_"+str(f_i)] = outputs_1["position_"+str(scale)]
                        outputs["prh_"+str(scale)+"_"+str(f_i)] = F.interpolate(
                            outputs["pr_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        
                        outputs["omaskb_"+str(scale)+"_"+str(f_i)],  outputs["omapb_"+str(scale)+"_"+str(f_i)]= self.get_occu_mask_backward(outputs["prh_"+str(scale)+"_"+str(f_i)])
                        outputs["omapbi_"+str(scale)+"_"+str(f_i)] = self.get_occu_mask_bidirection(outputs["ph_"+str(scale)+"_"+str(f_i)],
                                                                                                          outputs["prh_"+str(scale)+"_"+str(f_i)])

                    # Input for the AFNet
                    transform_input = [outputs["r_0"+"_"+str(f_i)], inputs[("color", 0, 0)]]
                    # Output from AFNet
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:

                        outputs["t_"+str(scale)+"_"+str(f_i)] = outputs_2[("transform", scale)]
                        outputs["th_"+str(scale)+"_"+str(f_i)] = F.interpolate(
                            outputs["t_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                        outputs["ref_"+str(scale)+"_"+str(f_i)] = (outputs["th_"+str(scale)+"_"+str(f_i)] * outputs["omaskb_"+str(scale)+"_"+str(f_i)].detach()  + inputs[("color", 0, 0)])
                        outputs["ref_"+str(scale)+"_"+str(f_i)] = torch.clamp(outputs["ref_"+str(scale)+"_"+str(f_i)], min=0.0, max=1.0)

                    # Input for PoseNet
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs["axisangle_0_"+str(f_i)] = axisangle
                    outputs["translation_0_"+str(f_i)] = translation
                    outputs["cam_T_cam_0_"+str(f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])

                    
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs["disp_"+str(scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs["depth_"+str(scale)] = depth

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs["cam_T_cam_0_"+str(frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs["axisangle_0_"+str(f_i)]
                    translation = outputs["translation_0_"+str(f_i)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs["sample_"+str(frame_id)+"_"+str(scale)] = pix_coords

                outputs["color_"+str(frame_id)+"_"+str(scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs["sample_"+str(frame_id)+"_"+str(scale)],
                    padding_mode="border")

                """print("Cam points")
                print(cam_points.shape)
                print("K")
                print(inputs[("K", source_scale)].shape)
                print("T")
                print(T.shape)
                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                print(outputs[("position_depth", scale, frame_id)][0].shape)
                print(outputs[("position_depth", scale, frame_id)][0])"""
                
    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0

        #outputs = outputs.reverse()
        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            #disp = outputs[("disp", scale,0)]
            #disp = outputs[scale]
            disp = outputs["disp_"+str(scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                
                occu_mask_backward = outputs["omaskb_"+str(0)+"_"+str(frame_id)].detach()
                
                loss_reprojection += (
                    self.compute_reprojection_loss(outputs["color_"+str(frame_id)+"_"+str(scale)], outputs["r_"+str(scale)+"_"+str(frame_id)]) * occu_mask_backward).sum() / occu_mask_backward.sum()
                loss_transform += (
                    torch.abs(outputs["r_"+str(scale)+"_"+str(frame_id)] - outputs["r_"+str(scale)+"_"+str(frame_id)].detach()).mean(1, True) * occu_mask_backward).sum() / occu_mask_backward.sum()
                    # self.compute_reprojection_loss(outputs[("refined", scale, frame_id)], outputs[("registration", 0, frame_id)].detach()) * occu_mask_backward).sum() / occu_mask_backward.sum()
                loss_cvt += get_smooth_bright(
                    outputs["th_"+str(scale)+"_"+str(frame_id)], inputs[("color", 0, 0)], outputs["r_"+str(scale)+"_"+str(frame_id)].detach(), occu_mask_backward)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0) 
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            #print(type(features))
            #print(len(features))
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs["r_"+str(scale)+"_"+str(frame_id)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:

                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs["th_"+str(s)+"_"+str(frame_id)][j].data, self.step)
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs["r_"+str(s)+"_"+str(frame_id)][j].data, self.step)
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs["ref_"+str(s)+"_"+str(frame_id)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs["omaskb_"+str(s)+"_"+str(frame_id)][j].data, self.step)

                writer.add_image("disp_{}/{}".format(s, j),normalize_image(outputs["disp_"+ str(s)][j]), self.step)
                    

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            save_path2 = os.path.join(save_folder, "{}.pt".format(model_name))
            to_save = model.state_dict()
            
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            
            torch.save(to_save, save_path)
            if model_name in ["encoder","depth","pose","pose_encoder"]: 
                print(model_name)
                sm = torch.jit.script(model)
                sm.save(save_path2)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        #save_path2 = os.path.join(save_folder, "{}.pt".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        #sm = torch.jit.script(self.model_optimizer.state_dict())
        #sm.save(save_path2)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            self.models[n].eval()
            for param in self.models[n].parameters():
                param.requires_grad = False

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")
    
    def pose_by_ransac(self, flow_2D, ref, target, intrinsic_inv_gpu,h_side, w_side, pose_gt=False, img_path=None):

        b, _, h, w = flow_2D.size()
        coord1_flow_2D, coord2_flow_2D = flow2coord(flow_2D)    # Bx3x(H*W) 
        coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
        coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)    
        margin = 10                 # avoid corner case


        E_mat = torch.zeros(b, 3, 3).cuda()                     # Bx3x3
        P_mat = torch.zeros(b, 3, 4).cuda()                     # Bx3x4

        PTS1=[]; PTS2=[];                                       # point list

        # process the frames of each batch
        for b_cv in range(b):
            # convert images to cv2 style
            if h_side is not None or w_side is not None:
                ref_cv =ref[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]
            else:
                ref_cv =ref[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
            ref_cv = (ref_cv*0.5+0.5)*255; tar_cv = (tar_cv*0.5+0.5)*255

            # detect key points           
            kp1, des1 = self.sift.detectAndCompute(ref_cv.astype(np.uint8),None)
            kp2, des2 = self.sift.detectAndCompute(tar_cv.astype(np.uint8),None)
            if len(kp1)<self.min_matches or len(kp2)<self.min_matches:
                # surf generally has more kps than sift
                kp1, des1 = self.surf.detectAndCompute(ref_cv.astype(np.uint8),None)
                kp2, des2 = self.surf.detectAndCompute(tar_cv.astype(np.uint8),None)

            try:
                # filter out some key points
                matches = self.flann.knnMatch(des1,des2,k=2)
                good = []; pts1 = []; pts2 = []
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
                
                # degengrade if not existing good matches
                if len(good)<self.min_matches:
                    good = [];pts1 = [];pts2 = []
                    for i,(m,n) in enumerate(matches):
                        good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
                pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2);
            except:
                # if cannot find corresponding pairs, ignore this sift mask 
                PTS1.append([None]); PTS2.append([None])

        assert len(PTS1)==b

        for batch in range(b):
            if cfg.SIFT_POSE:
                # if directly use SIFT matches
                pts1 = PTS1[batch]; pts2 = PTS2[batch]
                coord1_sift_2D = torch.FloatTensor(pts1)
                coord2_sift_2D = torch.FloatTensor(pts2)
                coord1_flow_2D_norm_i = torch.cat((coord1_sift_2D,torch.ones(len(coord1_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
                coord2_flow_2D_norm_i = torch.cat((coord2_sift_2D,torch.ones(len(coord2_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
            else:
                # check the number of matches
                if len(PTS1[batch])<self.min_matches or len(PTS2[batch])<self.min_matches:
                    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)
                    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)                
                else:
                    if cfg.SAMPLE_SP:
                        # conduct interpolation
                        pts1 = torch.from_numpy(PTS1[batch]).to(coord1_flow_2D.device).type_as(coord1_flow_2D)
                        B, C, H, W = coord1_flow_2D.size()
                        pts1[:,0] = 2.0*pts1[:,0]/max(W-1,1)-1.0;pts1[:,1] = 2.0*pts1[:,1]/max(H-1,1)-1.0
                        coord1_flow_2D_norm_i = F.grid_sample(coord1_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                        coord2_flow_2D_norm_i = F.grid_sample(coord2_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                    else:
                        # default choice
                        pts1 = np.int32(np.round(PTS1[batch]))
                        coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
                        coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

            intrinsic_inv_gpu_i = intrinsic_inv_gpu[batch].unsqueeze(0)

            # projection by intrinsic matrix
            coord1_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord1_flow_2D_norm_i) 
            coord2_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord2_flow_2D_norm_i) 
            # reshape coordinates            
            coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()
            coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()
            
            with autocast(enabled=False):
                # GPU-accelerated RANSAC five-point algorithm
                E_i, P_i, F_i,inlier_num = compute_P_matrix_ransac(coord1_flow_2D_norm_i.detach(), coord2_flow_2D_norm_i.detach(), 
                                                                intrinsic_inv_gpu[batch,:,:], self.delta, self.alpha, self.maxreps, 
                                                                len(coord1_flow_2D_norm_i), len(coord1_flow_2D_norm_i), 
                                                                self.ransac_iter, self.ransac_threshold) 

            E_mat[batch, :, :] = E_i.detach(); P_mat[batch, :, :] = P_i.detach()

        return P_mat, E_mat