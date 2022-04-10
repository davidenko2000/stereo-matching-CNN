device = 'cuda'

model = StereoCNN(in_channels=3)
model.to(device)

RGB_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4817, 0.5085, 0.5006), (0.3107, 0.3249, 0.3350))
	])
GRAY_transform = transforms.Compose([
	transforms.ToTensor(),
    	transforms.Normalize((0.4999), (0.3180))
	])

train_dataset = PatchesExtractor(tform=RGB_transform,train=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

criterion = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

train_report = np.zeros((EPOCHS, ))
num_batches = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    epoch_batches = 0
    model.train()
    for left_patch, right_pos_patch, right_neg_patch in train_dataloader:
      global num_batches
      num_batches += 1
      epoch_batches += 1
      left_patch, right_pos_patch, right_neg_patch = left_patch.to(device), right_pos_patch.to(device), right_neg_patch.to(device)
      left_output, right_pos_output, right_neg_output = model(left_patch), model(right_pos_patch), model(right_neg_patch)

      optimizer.zero_grad()
      loss = criterion(left_output, right_pos_output, right_neg_output)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

      if epoch_batches % 15000 == 0:
        print(f"Done {((BATCH_SIZE * epoch_batches) / len(train_dataset)) * 100:.3f} %")
        print('Train -> Loss: %.3f \n' % (train_loss/ epoch_batches))

    train_report[epoch] = train_loss / epoch_batches
    torch.save(model, f"train_model_{epoch}.pth")


for epoch in range(EPOCHS):
    train(epoch)
    if epoch == LR_CHANGE_AT_EPOCH:
            for param in optimizer.param_groups:
                  param['lr'] = LR_AFTER_10_EPOCHS

np.save(TRAIN_DATA + 'training_report.npy', train_report)
