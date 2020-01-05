# Bai nay minh hoa phep toan convolution 2D
# Su dung F.functional, padding
# http://cs231n.github.io/convolutional-networks/
from libcore import *

data = torch.tensor([
        [[1,2,0,1,2],
         [2,1,2,1,1],
         [2,0,2,2,1],
         [1,1,2,0,0],
         [0,1,0,0,1]
                ],
        [[2,0,1,2,1],
         [0,1,1,1,2],
         [1,2,1,2,0],
         [2,1,1,1,0],
         [2,1,2,1,1]
                ],
        [[2,0,1,0,0],
         [0,0,1,0,0],
         [0,1,2,1,0],
         [2,2,2,2,0],
         [2,0,0,0,1]
                ]
        ])

filters = torch.tensor([
        [[[-1.,-1, 1],
          [0,1,-1],
          [-1,0,1]
                ],
         [[-1,0,-1],
          [0,1,-1],
          [-1,-1,1]
                 ],
         [[1,0,-1],
          [-1,0,1],
          [1,0,0]
                 ]
                ],
        [[[-1,1,1],
          [0,1,-1],
          [-1,-1,0]
                ],
         [[0,-1,-1],
          [-1,0,-1],
          [1,-1,0]
                 ],
         [[-1,1,0],
          [1,-1,1],
          [-1,-1,-1]
                  ]
                ]
        ])

inputs = data[None].float()
bias = torch.tensor([1., 0.], dtype=torch.float32)

pad = F.pad(inputs, pad=(1,1,1,1))
result = F.conv2d(pad, filters, bias = bias, stride=2)

print(result)






