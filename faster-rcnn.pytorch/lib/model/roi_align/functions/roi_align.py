import torch
from torch.autograd import Function
from .._ext import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        #print("Hello, I am in forward function of ROI Align Function")
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.spatial_scale, features,
                                             rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.spatial_scale, features,
                                        rois, output)
#            raise NotImplementedError

        return output

    def backward(self, grad_output):
        #assert(self.feature_size is not None and grad_output.is_cuda)
        #print("Hello I am in backward function of ROI Align Function")
        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        #print("printing dir(roi_align)")
        print(dir(roi_align))
        #initially commented out roi_align_cuda       
        roi_align.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.spatial_scale, grad_output,
                                          self.rois, grad_input)
        #print("after back prop cuda function call")
        # print grad_input
        #roi_align.roi_align_backward(self.aligned_height,
        #                                  self.aligned_width,
        #                                  self.spatial_scale, grad_output,
        #                                  self.rois, grad_input)
        #print("after back prop function call")

        return grad_input, None
