
from typing import List, Union
import tensorflow as tf
import numpy as np
import math


def completeIou( boxGT, boxP ):

    #determine coordinates of the intersection rectangle
    xA = max( boxGT.min.x, boxP.min.x )
    yA = max( boxGT.min.y, boxP.min.y )
    xB = min( boxGT.max.x, boxP.max.x )
    yB = min( boxGT.max.y, boxP.max.y )

     #compute area of intersection rectangle
    interArea = max( 0, xB - xA ) * max( 0, yB - yA )
    
    #compute area of Bbox union
    areaGT = ( boxGT.max.x - boxGT.min.x ) * ( boxGT.max.y - boxGT.min.y )
    areaP = ( boxP.max.x - boxP.min.x ) * ( boxP.max.y - boxP.min.y )
    
    unionArea = areaGT + areaP - interArea
    
    iou = interArea / unionArea

    #Get diagonal between box centers

    centerGTX = ( boxGT.min.x + boxGT.max.x ) * 0.5
    centerGTY = ( boxGT.min.y + boxGT.max.y ) * 0.5
    centerPX = ( boxP.min.x + boxP.max.x ) * 0.5
    centerPY = ( boxP.min.y + boxP.max.y ) * 0.5

    distX = abs( centerPX - centerGTX )
    distY = abs( centerPY - centerGTY )


    #Get diagonal for enclosing box
    exA = min( boxGT.min.x, boxP.min.x ) * 1.0
    eyA = min( boxGT.min.y, boxP.min.y ) * 1.0
    exB = max( boxGT.max.x, boxP.max.x ) * 1.0
    eyB = max( boxGT.max.y, boxP.max.y ) * 1.0

    eDistX = abs( exA - exB )
    eDistY = abs( eyA - eyB )

    centerDis = math.sqrt( distX**2 + distY**2 )
    encloseDis = math.sqrt( eDistX**2 + eDistY**2 )

    diou = 1 - iou + ( centerDis**2 / encloseDis**2 )

    print( "Diou: " + str(diou) + " centerDistance: " + str(centerDis) + " encloseDis: " + str(encloseDis) + " iou: " + str(iou) )
    return ( int(exA), int(eyA), int(exB), int(eyB), int(centerGTX), int(centerGTY), int(centerPX), int(centerPY) )

def diouCoef( boxGT, boxP ):

    half = tf.convert_to_tensor(0.5, dtype = tf.float32 )
    center_vec = tf.math.abs( ( boxGT[..., :2] + boxGT[..., 2:] ) * half - ( boxP[..., :2] + boxP[..., 2:] ) * half )

    boxGT = tf.concat([boxGT[..., :2] - boxGT[..., 2:] * 0.5,
                        boxGT[..., :2] + boxGT[..., 2:] * 0.5], axis=-1)
    boxP = tf.concat([boxP[..., :2] - boxP[..., 2:] * 0.5,
                        boxP[..., :2] + boxP[..., 2:] * 0.5], axis=-1)

    boxGT = tf.concat([tf.minimum(boxGT[..., :2], boxGT[..., 2:]),
                        tf.maximum(boxGT[..., :2], boxGT[..., 2:])], axis=-1)
    boxP = tf.concat([tf.minimum(boxP[..., :2], boxP[..., 2:]),
                        tf.maximum(boxP[..., :2], boxP[..., 2:])], axis=-1)

    boxes1_area = (boxGT[..., 2] - boxGT[..., 0]) * (boxGT[..., 3] - boxGT[..., 1])
    boxes2_area = (boxP[..., 2] - boxP[..., 0]) * (boxP[..., 3] - boxP[..., 1])

    left_up = tf.maximum(boxGT[..., :2], boxP[..., :2])
    right_down = tf.minimum(boxGT[..., 2:], boxP[..., 2:])

    zeros = tf.convert_to_tensor(0.0, dtype = tf.float32 )
    inter_section = tf.maximum( right_down - left_up, zeros )
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxGT[..., :2], boxP[..., :2])
    enclose_right_down = tf.maximum(boxGT[..., 2:], boxP[..., 2:])
    enclose_vec = enclose_right_down - enclose_left_up
    center_dis = tf.sqrt( center_vec[..., 0]**2 + center_vec[..., 1]**2 )
    enclose_dis = tf.sqrt( enclose_vec[..., 0]**2 + enclose_vec[..., 1]**2 )

    diou = 1.0 - iou + ( center_dis**2 / enclose_dis**2 )

    return diou

'''
def ciouCoef( target_boxes: FloatType, pred_boxes: FloatType ) -> tf.Tensor:
     
     t_ymin = target_boxes[..., 0]
     t_xmin = target_boxes[..., 1]
     t_ymax = target_boxes[..., 2]
     t_xmax = target_boxes[..., 3]

     p_ymin = pred_boxes[..., 0]
     p_xmin = pred_boxes[..., 1]
     p_ymax = pred_boxes[..., 2]
     p_xmax = pred_boxes[..., 3]

     zero = tf.convert_to_tensor(0.0, t_ymin.dtype)
     p_width = tf.maximum(zero, p_xmax - p_xmin)
     p_height = tf.maximum(zero, p_ymax - p_ymin)
     t_width = tf.maximum(zero, t_xmax - t_xmin)
     t_height = tf.maximum(zero, t_ymax - t_ymin)
     p_area = p_width * p_height
     t_area = t_width * t_height

     intersect_ymin = tf.maximum(p_ymin, t_ymin)
     intersect_xmin = tf.maximum(p_xmin, t_xmin)
     intersect_ymax = tf.minimum(p_ymax, t_ymax)
     intersect_xmax = tf.minimum(p_xmax, t_xmax)
     intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
     intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
     intersect_area = intersect_width * intersect_height

     union_area = p_area + t_area - intersect_area
     iou_v = tf.math.divide_no_nan(intersect_area, union_area)

     enclose_ymin = tf.minimum(p_ymin, t_ymin)
     enclose_xmin = tf.minimum(p_xmin, t_xmin)
     enclose_ymax = tf.maximum(p_ymax, t_ymax)
     enclose_xmax = tf.maximum(p_xmax, t_xmax)

     p_center = tf.stack([(p_ymin + p_ymax) / 2, (p_xmin + p_xmax) / 2])
     t_center = tf.stack([(t_ymin + t_ymax) / 2, (t_xmin + t_xmax) / 2])
     euclidean = tf.linalg.norm(t_center - p_center)
     diag_length = tf.linalg.norm([enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin])

     diou_v = iou_v - tf.math.divide_no_nan(euclidean**2, diag_length**2)

     v = _get_v(p_height, p_width, t_height, t_width)
     alpha = tf.math.divide_no_nan(v, ((1 - iou_v) + v))

     return diou_v - alpha * v  


def _get_v( b1_height: FloatType, b1_width: FloatType, b2_height: FloatType, b2_width: FloatType ) -> tf.Tensor:


  @tf.custom_gradient
  def _get_grad_v(height, width):
    """backpropogate gradient."""
    arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
        tf.math.divide_no_nan(width, height))
    v = 4 * ((arctan / math.pi)**2)

    def _grad_v(dv):
      """Grad for eager mode."""
      gdw = dv * 8 * arctan * height / (math.pi**2)
      gdh = -dv * 8 * arctan * width / (math.pi**2)
      return [gdh, gdw]

    def _grad_v_graph(dv, variables):
      """Grad for graph mode."""
      gdw = dv * 8 * arctan * height / (math.pi**2)
      gdh = -dv * 8 * arctan * width / (math.pi**2)
      return [gdh, gdw], tf.gradients(v, variables, grad_ys=dv)

    if tf.compat.v1.executing_eagerly_outside_functions():
      return v, _grad_v
    return v, _grad_v_graph

  return _get_grad_v(b2_height, b2_width)
  '''