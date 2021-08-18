## Difference between *nn.modules* vs *nn.children*
```python
class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.convBN = nn.Sequential(nn.Conv2d(10,10,3), nn.BatchNorm2d(10))
    self.linear = nn.Linear(10,2)
    
  def forward(self, x):
    pass
    
Net = myNet()

print("Printing children\n----------------------")
print(list(Net.children()))
print(Printing Modules\n------------------------")
print(list(Net.modules()))
```

![image](https://user-images.githubusercontent.com/38284936/129830944-0e47f3c7-3eb2-4fc3-a75a-a7fe8f7e0c28.png)


