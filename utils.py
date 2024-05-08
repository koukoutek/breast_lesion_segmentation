import numpy as np
import math
import os
import torch
import xml.etree.ElementTree as ET
import re
import cc3d

from pathlib import Path
from skimage.draw import polygon
from monai.transforms import MapTransform
from monai.networks.nets import UNet


class IndexTracker:
    def __init__(self, ax, X, vmin, vmax):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.vmin = vmin
        self.vmax = vmax

        self.im = ax.imshow(self.X[:, :, self.ind], vmax=self.vmax, vmin=self.vmin, cmap='gray') #cmap='gray',
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step)) # print step and direction
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class Annotation:
      
  def __init__(self, xml_path, filename, shape):

    self.xml_path = xml_path + filename + '.xml'
    self.mask     = self.create_mask_array(shape)
    self.fill_mask()


  """
  This function creates the proper contour/polygon for each ROI and stores
  the information in the corresponding position of the mask object
  """
  def fill_mask(self):
    if os.path.exists(self.xml_path):
      rois, num_rois   = self.parse_XML(self.xml_path)

      for roi in rois:
        roi_info       = self.get_roi_info(roi)
        r_poly, c_poly = self.create_polygon_lists(self.mask, roi_info["points"])
        rr, cc         = polygon(r_poly, c_poly)
        roi_channel    = self.select_mask_channel(roi_info["roi_type"])
        try:
          self.mask[rr, cc, roi_channel] = 1
        except IndexError:
          print(self.xml_path)


  """
  Inputs:
    -- xml_path: Path to the corresponding xml file
  Outputs:
    -- rois: Array with the ROI objects
    -- num_of_rois: Number of ROIs 
  """
  def parse_XML(self, xml_path):
    tree        = ET.parse(xml_path)
    root        = tree.getroot()       # The root of the XML file
    data        = root[0][1]           # The essential info
    rois        = data[0][5]           # Array containing the ROI objects
    num_of_rois = int(data[0][3].text) # Number of ROI objects

    return rois, num_of_rois


  """
  Inputs:
    -- img_shape: The preferred shape of the mask to be created
  Outputs:
    -- 3-dimensional numpy array of type uint8 
  """
  def create_mask_array(self,img_shape):
    return np.zeros((img_shape[0], img_shape[1], 3), dtype = np.uint8)

  def get_roi_info(self, roi):
    roi_info      = {
      "points":        roi[21],           # Array containing the points of a ROI
      "num_of_points": int(roi[17].text), # Number of points of the area
      "roi_index":     int(roi[7].text),  # Identifier of the ROI
      "roi_type":      roi[15].text       # (Mass, Calcification, other)
    }

    return roi_info



  """
  Inputs:
    -- mask: numpy object of the mask
    -- points: x-y coordinates of a ROI's points
  Outputs:
    -- r_poly: array containing the x-axis coordinates
    -- c_poly: array containing the y-axis coordinates
  """
  def create_polygon_lists(self, mask, points):
    mask_width  = mask.shape[0]
    mask_height = mask.shape[1]
    r_poly      = np.array([])
    c_poly      = np.array([])
    roi_img     = np.zeros((mask_width, mask_height), dtype=np.uint8)

    for point in points:

      temp_tuple = point.text[1:-1].split(",")
      y          = int(math.trunc(float(temp_tuple[0]))) 
      x          = int(math.trunc(float(temp_tuple[1])))
      r_poly     = np.append(r_poly, x)
      c_poly     = np.append(c_poly, y)

    return r_poly, c_poly


  """
  Input:
    -- roi_type: The type of a specific ROI, extracted from the XML file
  Output:
    -- roi_channel: The type of the ROI defines the integer value of this var
  """
  def select_mask_channel(self, roi_type):
    roi_ch = 2
    if roi_type == "Mass":
      roi_ch = 0
    elif roi_type == "Calcification":
      roi_ch = 1
    return roi_ch
  

class ConvertINBreastLesionToMultiChannelMaskd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        
        background = d['mask'][0] == 0
        lesion = d['mask'][0] == 1

        d['segmentation'] = np.stack((background, lesion), axis=0)

        return d
    
  
class WindowindINBreastImageBasedOnPercentiled(MapTransform):

    def __call__(self, data):
        d = dict(data)

        im_copy = d['image'].clone()
        # windowing should be based on the energy level
        upper_value = np.percentile(im_copy, 99)
        lower_value = np.percentile(im_copy, 5)

        # d['image'] = torch.clip(d['image'], lower_value, upper_value)
        d['image'] = (d['image'] - d['image'].min())/(d['image'].max() - d['image'].min())

        d['image_shape'] = d['image'][0].shape

        return d


class LoadMSynthMaskd(MapTransform):
    """
        Load the mask from the MSynth dataset.
    """
    def __call__(self, data):
        d = dict(data)
        d['mask'] = str(d['mask'])
        data = read_mhd(d['mask'])   
        pixel_array = np.fromfile(d["mask"].replace('mhd', 'raw'), dtype="float32").reshape(
            data["NDims"], data["DimSize"][1], data["DimSize"][0]
        )
        tmp = pixel_array[0]
        X = np.std(tmp) * 2
        TH = np.mean(tmp)
        tmp[tmp < TH - X] = 0
        tmp[tmp > TH + X] = X + TH
        tmp = tmp > 4

        tmp = cc3d.largest_k(tmp.astype(np.float32), k=1, connectivity=8)

        d['mask'] = np.expand_dims(tmp, axis=0).astype(np.float32)      
        return d
    

class LoadMSynthImaged(MapTransform):
    """
        Load the image from the MSynth dataset.
    """
    def __call__(self, data):
        d = dict(data)
        d['image'] = str(d['image'])
        data = read_mhd(d['image'])   
        pixel_array = np.fromfile(d["image"].replace('mhd', 'raw'), dtype="float32").reshape(
            data["NDims"], data["DimSize"][1], data["DimSize"][0]
        )
        tmp = pixel_array[0]
        X = np.std(tmp) * 3 # 4 # 3 # 2
        TH = np.mean(tmp)
        tmp[tmp < TH - X] = 0
        tmp[tmp > TH + X] = X + TH
        d['image'] = np.expand_dims(tmp, axis=0).astype(np.float32)
        return d
    

class WindowMSynthImaged(MapTransform):

    def __call__(self, data):
        d = dict(data)
        d['image'] = (d['image'] - np.amin(d['image']))/np.ptp(d['image'])
        d['image'][d['image'] < .05] = 0
        d['image_shape'] = d['image'][0].shape
        return d
    

class ConvertMSynthLesionToMultiChannelMaskd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        background = d['mask'][0] == 0
        lesion = d['mask'][0] == 1
        d['segmentation'] = np.stack((background, lesion), axis=0)
        return d
    

def read_mhd(filename):
    if 'None' in filename:
        return None
    data = {}
    with open(filename, "r") as f:
        for line in f:
            s = re.search("([a-zA-Z]*) = (.*)", line)
            data[s[1]] = s[2]

            if " " in data[s[1]]:
                data[s[1]] = data[s[1]].split(" ")
                for i in range(len(data[s[1]])):
                    if data[s[1]][i].replace(".", "").replace("-", "").isnumeric():
                        if "." in data[s[1]][i]:
                            data[s[1]][i] = float(data[s[1]][i])
                        else:
                            data[s[1]][i] = int(data[s[1]][i])
            else:
                if data[s[1]].replace(".", "").replace("-", "").isnumeric():
                    if "." in data[s[1]]:
                        data[s[1]] = float(data[s[1]])
                    else:
                        data[s[1]] = int(data[s[1]])
    return data
    
    
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
    

def evaluate_true_false(inp):
    inp = str(inp).upper()
    if 'TRUE'.startswith(inp):
        return True
    elif 'FALSE'.startswith(inp):
        return False
    else:
        raise ValueError('Argument error. Expected bool type.')


def read_model(model_path):
    # Only used for evaluation script
    saved_model = torch.load(model_path)

    # 1-channel input
    model = UNet(spatial_dims=2, in_channels=1, out_channels=2, kernel_size=3, up_kernel_size=3, channels=[32, 64, 128, 256, 512, 1024],
                        strides=[2, 2, 2, 2, 2], norm='instance', dropout=.4, num_res_units=3)
    model_dict = saved_model['model_state_dict']

    new_dict = {}
    for k,v in model_dict.items():
        if str(k).startswith('module'): # module will be there in case of training in multiple GPUs
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    model.load_state_dict(new_dict)
    
    return model
    
    

    



    

