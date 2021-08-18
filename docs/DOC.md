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

## Working with *collate_fn*
The use of *collate_fn* is slightly different when automatic batching is enabled or disabled.

* When automatic batching is disabled: *collate_fn* is called with each individual data sample, and the output is yielded from the data loader iterator. In this case, the default *collate_fn* simply converts NumPy arrays in PyTorch tensors.

* When automatic batching is enabled: *collate_fn* is called with a list of data samples at each time. It is expected to collate the input samples into a batch for yielding from the data loader iterator. The rest of this section describes behavior of the default *collate_fn* in this case.

For instance, if each data sample consists of a 3-channel image and an integral label, i.e., each element of the dataset returns a tuple *(image, class_index)*, the default *collate_fn* collates a list of such tuples into a single tuple of a batched image tensor and a batched class label Tensor. In particular, the default *collate_fn* has the following properties:
- It always prepends a new dimension as the batch dimension
- It automatically converts NumPy arrays and Python numerical values into PyTorch Tensors.
- It preserves the data structure, e.g., if each sample is a dictionary, it outputs a dictionary with the same set of keys but batched Tensors as values (or lists if the values cannot be converted into Tensors). 

Users may use customized *collate_fn* to achieve custom batching, e.g., collate along a dimension rather than the first, padding sequences of various lengths, or adding support for custom data types.

## Tensor Operation for Deep Learning: concatenate VS stack
The difference between *concatenating* and *stacking* tensors can be described in a single sentence.

> Concatenating joins a sequence of tensors along an existing axis, and stacking joins a sequence of tensors along a new axis. 

For the most part, concatenating along an existing axis of a tensor is pretty straightforward. The confusion usually arises when we want to stack along a new axis. Another way of saying that we stack is to say that we create a new axis and then concat on that axis. 

### How to add or insert an axis into a Tensor
```python
import torch
t1 = torch.tensor([1,1,1])
```
Here, we're importing PyTorch and creating a simple tensor that has a single axis of length 3. Now, to add an axis to a tensor in PyTorch, we use the *unsqueeze()* function. Note that this is the opposite of squeezing.

```python
t1.unsqueeze(dim=0)
> tensor([[1,1,1]])
```

Here, we are adding an axis, a.k.a dimension at index zero of this tensor. This gives us a tensor with shape of *1x3*. When we say index zero of the tensor, we mean the first index of the tensor's shape.

Now, we can also add an axis at the second index of this tensor.
```python
t1.unsqueeze(dim=1)
> tensor([[1],
	[1],
	[1]])
```

This gives us a tensor with a shape of *3x1*. Adding axes like this changes the way the data is organized inside the tensor, but it does not change the data itself. Basically, we are just reshaping the tensor. We can see that by checking the shape of each one of these.

```python
print(t1.shape)
print(t1.unsqueeze(dim=0).shape)
print(t1.unsqueeze(dim=1).shape)
> torch.Size([3])
> torch.Size([1,3])
> torch.Size([3,1])
```

Now, thinking back about concatenating versus stacking, when we concat, we are joining a sequence of tensors along an existing axis. This means that we are extending the length of an existing axis.

When we stack, we are creating a new axis that didn't exist before and this happens across all the tensors in our sequence, and then we concat along this new sequence.

### stack VS cat in PyTorch
```python
t1 = torch.tensor([1,1,1])
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])
```
Now, let's concatenate these with one another. Notice that each of these tensors have a single axis. This means that the result of the `cat` function will also have a single axis. This is because when we concatenate, we do it along an existing axis. Notice that in this example, the only existing axis is the first axis.
```python
torch.cat((t1,t2,t3),dim=0)
> tensor([1,1,1,2,2,2,3,3,3])
```
Alright, we took three single axis tensors each having an axis length of three, and now we have a single tensor with an axis length of nine.

Now, let's stack these tensors along a new axis that we'll insert. We'll insert an axis at the first index. Note that this insertion will be happening implicitly under the hood by the `stack` function.
```python
torch.stack((t1,t2,t3), dim=0)
> tensor([[1,1,1],
	[2,2,2],
	[3,3,3]])
```

This gives us a new tensor that has a shape of *3x3*. Notice how the three tensors are concatenated along the first axis of this tensor. Note that we can also insert the new axis explicitly, and perform the concatenation directly.

```python
torch.cat(
	(t1.unsqueeze(0),
	t2.unsqueeze(0),
	t3.unsqueeze(0)
	), dim=0)
> tensor ([[1,1,1],
	  [2,2,2],
	  [3,3,3]])
```
Note that we cannot concat this sequence of tensors along the second axis because there currently is not second axis in existence, so in this case, stacking is our only option.

```python
torch.stack((t1,t2,t3),dim=1)
torch.cat(
	(t1.unsqueeze(1),
	t2.unsqueeze(1),
	t3.unsqueeze(1)), dim=1)
> tensor([[1,2,3],
	[1,2,3],
	[1,2,3]])
```
### Stack or Concat: Real-Life Examples
Here are three concrete examples that we can encounter in real life. Let's decide when we need to stack and when we need to concat.

#### Joining Images Into A Single Batch
Suppose we have three individual images as tensors. Each image tensor has three dimensions, a channel axis, a height axis, a width axis. Note that each of these tensors are separate from one another. Now, assume that our task is to join these tensors together to form a single batch tensor of three images.

Do we concat or do we stack?

Well, notice that in this example, there are only three dimensions in existence,and for a batch, we need four dimensions. This means that the answer is to stack the tensors along a new axis. This new axis will be the batch axis. This will give us a single tensor with four dimensions by adding one for the batch.

Note that if we join these three along any of the existing dimensions, we would be messing up either the channels, the height, or the width. We don't want to mess our data up like that.

```python
import torch
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)

torch.stack((t1,t2,t3), dim=0).shape

> torch.Size([3,3,28,28])

torch.concatenate((t1.unsqueeze(0),
		t2.unsqueeze(0),
		t3.unsqueeze(0)), dim=0).shape

> torch.Size([3,3,28,28])
```
#### Joining Batches Into a Single Batch
Now, suppose we have the same three images as before, but this time the images already have a dimension for the batch. This actually means that we have three batches of size one. Assume that it is our task to obtain a single batch of three images.

Do we concat or do we stack?

Well, notice how there is an existing dimension that we can concat on. This means that we concat these along the batch dimension. In this case there is no need to stack.

```python
import torch
t1 = torch.zeros(1,3,28,28)
t2 = torch.zeros(1,3,28,28)
t3 = torch.zeros(1,3,28,28)
torch.cat((t1,t2,t3),dim=0).shape
> torch.Size([3,3,28,28])
```

#### Joining Images With an Existing Batch
Suppose we have the same three separate image tensors. Only this time, we already have a batch tensor. Assume our task is to join these three separate images with the batch. 

Do we concat or do we stack?

Well, notice how the batch axis already exists inside the batch tensor.
However, for the images, there is no batch axis in existence. This means neither of these will work. To join with stack or cat, we need tensors to have matchingshapes. So then, are we stuck? Is this impossible?

It is indeed possible. It's actually a very common task. The answer is to first stack and then to concat.

We first stack the three image tensors with respect to the first dimension. This creates a new batch dimension of length three. Then, we can concat this new tensor with the batch tensor.

```python
import torch
batch = torch.zeros(3,3,28,28)
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)

torch.cat((batch, torch.stack((t1,t2,t3), dim=0)), dim=0).shape

> torch.Size([6,3,28,28])
```

this is the same as the below.

```python
import torch
batch = torch.zeros(3,3,28,28)
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)

torch.cat((batch,t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)),dim=0).shape

> torch.Size([6,3,28,28])
```



