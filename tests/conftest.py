import torch

# Testlerdeki modeller çok küçük; bu boyutta çok iş parçacığı hızlandırmak
# yerine yavaşlatıyor (iş parçacıklarını eşitleme maliyeti hesaptan büyük).
torch.set_num_threads(1)
