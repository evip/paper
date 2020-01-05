# Bai nay minh hoa phep toan convolution 2D
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

#pad = F.pad(inputs, pad=(1,1,1,1))
model = torch.nn.Conv2d(3,2, kernel_size=(3,3),stride=2, padding=1)

# 1. Kieu du lieu khoi tao phai la float32
# 2. Neu khong dung weight.data thi phair dung nn.Parameters
# 3. Neu muon dong bang tang 1 thi dung requires_grad = False
model.weight = torch.nn.Parameter(filters, requires_grad=False)
model.bias = torch.nn.Parameter(torch.tensor([1., 0.], dtype=torch.float32), requires_grad=False)

result = model(inputs)
print(result)


