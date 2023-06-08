import torchvision.models as models
import torch.nn as nn

def build_model(version='b0', pretrained=True, fine_tune=True, num_classes=2):
    #az osztályok száma legyen mindig 2, még a teszthalmazon is, mert arra vagyunk kiváncsiak, hogyan osztályozná az ismeretlen képeket
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
        
    if version=='b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    elif version=='b1':
        model = models.efficientnet_b1(pretrained=pretrained)
    elif version=='b2':
        model = models.efficientnet_b2(pretrained=pretrained)
    elif version=='b3':
        model = models.efficientnet_b3(pretrained=pretrained)
    elif version=='b4':
        model = models.efficientnet_b4(pretrained=pretrained)
    elif version=='b5':
        model = models.efficientnet_b5(pretrained=pretrained)
    elif version=='b6':
        model = models.efficientnet_b6(pretrained=pretrained)
    elif version=='b7':
        model = models.efficientnet_b7(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    #in all version of effnet, the outputfeatures of the last layer is 1000. We need to change that
    model.classifier = nn.Sequential(nn.Dropout2d(p =model.classifier[0].p, inplace=True),
                            nn.Linear(in_features= model.classifier[1].in_features, out_features=model.classifier[1].out_features),
                            nn.Linear(in_features= model.classifier[1].out_features, out_features=num_classes))
    return model