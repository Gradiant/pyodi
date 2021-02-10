# anchor_generator=dict(
#     type='AnchorGenerator',
#     scales=[1.12, 3.15, 8.12],
#     ratios=[0.33, 0.67, 1.40],
#     strides=[4, 8, 16, 32, 64],
#     base_sizes=[4, 8, 16, 32, 64],
# )


anchor_generator=dict(
    type='AnchorGenerator',
    scales=[1.1285421704860277, 3.152189116182455, 8.120843227784663],
    ratios=[0.34290467231587757, 0.6834987264342319, 1.4248827778391429],
    strides=[4, 8, 16, 32, 64],
    base_sizes=[4, 8, 16, 32, 64],
)