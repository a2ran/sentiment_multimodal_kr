# sentiment_multimodal_kr
사용자의 음성 데이터를 받아 speech/text 두가지 항목을 반영해 classify한 multimodal 모델입니다.

## 07/10

First Draft 업로드

#### 사용 데이터 : 

`감정 분류를 위한 대화 음성 데이터셋` : [https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata)
* 일정 기간동안 사용자들이 어플리케이션과 자연스럽게 대화하고, 수집된 데이터를 정제 작업을 거쳐 선별
* 7가지 감정(happiness, angry, disgust, fear, neutral, sadness, surprise)에 대해 5명이 라벨링

#### speech_classification_bert

* `감정 분류를 위한 대화 음성 데이터셋`의 wav 파일과 연동된 text을 가지고 다섯가지 category에 대한 text sentiment analysis 진행.
```
array(['anger', 'disgust', 'fear', 'neutral', 'sad'], dtype=object)
```
* classification 모델로는 'distilbert-base-multilingual-cased' 사용, 프레임워크로 `PyTorch Lightning 사용`

```
class Classifier(pl.LightningModule):
    def __init__(self,num_classes=5):
        super().__init__()
        self.model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.classifier = torch.nn.Linear(768, num_classes)
        self.preds = np.array([])
        self.labels = np.array([])
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task="multiclass")
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task="multiclass")

    def forward(self, x):
        x = self.model(**x)
        x = x.last_hidden_state[:,0]
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        self.log('train_acc', self.train_acc(y_hat.argmax(dim=1), y), prog_bar = True)
        self.log('train_loss', loss, prog_bar = True)
        return loss
```
* Parameters :
```
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1,total_iters=100),
                'interval': 'step',
            },
            'monitor': 'val_recall',
            'interval': 'epoch'
        }

    trainer = pl.Trainer(
        devices= "auto",
        accelerator='gpu',
        logger=logger,
        max_epochs=5,
        callbacks=[checkpoints],
        precision=16,
    )
```

![image](https://github.com/a2ran/sentiment_multimodal_kr/assets/121621858/40b0c466-b656-4093-b067-d5e200c71c79)

#### speech_audio_classification

* `감정 분류를 위한 대화 음성 데이터셋`의 wav 파일을 가지고 Wav2VecFeatureExtractor으로 음성 파일에 대한 5가지 category에 대한 classificaiton 진행
```
class AudioModel(pl.LightningModule):
    def __init__(self,num_classes, ckpt='kresnik/wav2vec2-large-xlsr-korean'):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(ckpt)
        self.model.feature_extractor._freeze_parameters()
        self.layer_weights = torch.nn.Parameter(torch.ones(25))
        self.linear = torch.nn.Linear(1024*2, num_classes)
        self.dropout = torch.nn.Dropout(0.2)
        self.preds = []
        self.labels = []

    def compute_features(self, x):
        x = self.model(input_values=x, output_hidden_states=True).hidden_states
        x = torch.stack(x,dim=1)
        weights = torch.nn.functional.softmax(self.layer_weights, dim=-1)
        mean_x = x.mean(dim = 2)
        std_x = x.std(dim = 2)
        x = torch.cat((mean_x, std_x), dim=-1)
        x = (x * weights.view(-1,25,1)).sum(dim=1)
        return x

    def forward(self, x):
        x = self.compute_features(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = torch.softmax(x,dim=-1)
        return x

    def training_step(self, batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits,y)
        self.log('train_loss', loss,sync_dist=True)
        return loss
```

### Multimodal
* text model의 best param과 speech model의 best param을 합쳐 최적의 멀티모달 모델 생성
```
class LateFusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.audio_model = AudioModel.load_from_checkpoint('./trained/speech_best.ckpt',num_classes=5)
        self.text_model = Classifier.load_from_checkpoint('./trained/text_best.ckpt')
        self.proj = torch.nn.Linear(773,512)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(512, 5)
        self.preds = np.array([])
        self.labels = np.array([])
        self.train_acc = torchmetrics.Accuracy(num_classes=5, average='macro', task="multiclass")
        self.valid_acc = torchmetrics.Accuracy(num_classes=5, average='macro', task="multiclass")

    def freeze(self):
        for param in self.audio_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, audio_input, text_input):
        audio_emb = self.audio_model.forward(audio_input)
        text_emb = self.text_model.model(**text_input)
        text_emb = text_emb.last_hidden_state[:,0]
        x = torch.cat([audio_emb,text_emb], dim=1)
        x = self.dropout(x)
        x = self.proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        audio_input, text_input, y = batch
        y_hat = self(audio_input, text_input)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        self.log('train_acc', self.train_acc(y_hat.argmax(dim=1), y), prog_bar = True)
        self.log('train_loss', loss, prog_bar = True)
        return loss
```

![image](https://github.com/a2ran/sentiment_multimodal_kr/assets/121621858/973a6a69-c420-4bb1-a135-0d7fc7df4b34)

##### accuracy : 0.4693 -> 0.5818
