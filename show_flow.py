import cv2
import flow as flib

flow = flib.readFlow('./tvl1.flo')  # H, W, 2
# print(flow.shape)
print(flow[:, :, 0])
print(flow.shape)
# print(flow[410-5:410+5, 556-5:556+5, 1])
flow_color = flib.flow_to_color(flow, clip_flow=None, convert_to_bgr=True)  # H, W, 3
cv2.imwrite('test_tvl1.png', flow_color)