from timeit import default_timer as timer

import torch

def train(net, training_set, lr=0.001, momentum=0.9):
  print("Training started with %d images ..." % len(training_set))
  start = timer()
  
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  
  for _ in range(2):
    for inputs, labels in training_set:
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  
  end = timer()
  print("Training finished after %.2f seconds" % (end - start))
