import torch

def predict(net, image):
  return torch.max(net(image), 1)[1]

def validate(net, validation_set):
  n_validation = len(validation_set)
  print("Validating with %d images" % n_validation)
  
  n_errors = 0
  for i, data in enumerate(validation_set, 0):
    image, label = data
    output = net(image)
    predicted = torch.max(output, 1)[1]
    if not torch.eq(predicted, label):
      n_errors += 1
  
  accuracy = 1 - n_errors / n_validation
  print("The accuracy is %.2f%%" % (100*accuracy))
