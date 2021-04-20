"""
Neural Network library for Parts of Speech Tagging Tweets
by Matt Mauer
"""

START = "<s>"
STOP = "</s>"

def build_context_window(data, center, w=1):
    leftX = []
    rightX = []
    left_end = False
    right_end = False
    
    for i in range(1, w+1):
        
        # build left of center
        leftI = center - i
        if left_end:
            leftX.append(START)
        elif leftI < 0:
            leftX.append(START)
            left_end = True
        elif not data[leftI][0]:
            leftX.append(START)
            left_end = True
        else:
            leftX.append(data[leftI][0])
            
        # build right of center
        rightI = center + i
        if right_end:
            rightX.append(STOP)
        elif center+i+1 > len(data):
            rightX.append(STOP)
            right_end = True
        elif not data[rightI][0]:
            rightX.append(STOP)
            right_end = True
        else:
            rightX.append(data[rightI][0])
            
    X = leftX[::-1] + [data[center][0]] + rightX
    return X


class FeedForwardH1Net(nn.Module):
    
    def __init__(self, V=30000, window_size=0, embedding_dim=50, num_labels=25):
        super(FeedForwardH1Net, self).__init__()
        self.embedding = nn.Embedding(V, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1), 128)
        self.linear2 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.embedding.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs
    
    
class BaselinePOSTagger:
    
    def __init__(self, word_to_ix={}, pos_to_ix={}, pos_from_ix={}, unk_ix=set(), window_size=0, lr=0.01, default_net=True):
        self.word_to_ix = word_to_ix
        self.unk_ix = unk_ix
        self.pos_to_ix = pos_to_ix
        self.pos_from_ix = pos_from_ix
        self.window_size = window_size
        
        if default_net:
            self.net = FeedForwardH1Net(V=len(word_to_ix), window_size=window_size, num_labels=len(pos_to_ix))
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
        self.loss_F = nn.NLLLoss()
        self.training_time = 0
        self.dev_score = 0
        self.test_score = 0
        self.peak_epoch = 0
        
    def get_word_index(self, w, training=True):
        
        seen = w in self.word_to_ix
        
        if seen and training and w in self.unk_ix:
            coin_flip = random.randint(0,1)
            if coin_flip:
                return self.word_to_ix[w]
            else:
                return self.word_to_ix["UUUNKKK"]
        elif seen:
            return self.word_to_ix[w]
        else:
            return self.word_to_ix["UUUNKKK"]
    
    def get_features(self, x, training=True):
        
        embed_ixs = [self.get_word_index(w, training) for w in x]
        inputs = torch.tensor(embed_ixs, dtype=torch.long)
        return inputs
    
    def train(self, XYtrain, XYdev, XYtest, epochs=1, verbose=True):
        
        start = time.time()
        
        # for each epoch
        for epoch in range(1, epochs+1):
        
            i = 0
            while i < len(XYtrain):
                i += 1
                
                x, y = random.choice(XYtrain)
                
                # construct net input — turn data into tensors NEEDs WORK
                inputs = self.get_features(x)
                
                # reset gradients to zero
                self.net.zero_grad()
                
                # feed forward
                probs = self.net(inputs)
                
                # compute loss -- NEEDS WORK: 
                loss = self.loss_F(probs, torch.tensor([self.pos_to_ix[y]], dtype=torch.long))
                
                # backprop and update weights
                loss.backward()
                self.optimizer.step()
            
            if verbose:
            
                # test on dev data
                dev_score = self.test(XYdev)
                print(f"After epoch {epoch}, accuracy is {round(dev_score, 4)} on the DEV data.")
                
                if dev_score > self.dev_score:
                    self.dev_score = dev_score
                    test_score = self.test(XYtest)
                    print(f"After epoch {epoch}, accuracy is {round(test_score, 4)} on the DEVTEST data.")
                    
                    if test_score > self.test_score:
                        self.test_score = test_score
                        self.peak_epoch = epoch
                
        self.training_time += time.time() - start
        
        if verbose:
            print("\n------------------------------------------------------------------\n")
            print(f"{round(self.training_time, 1)} seconds has been spent training.")
            print(f"The best score on DEVTEST data was {round(self.test_score, 4)} after epoch {self.peak_epoch}")
                
                    
    def predict(self, x):
        
        inputs = self.get_features(x, training=False)
        # torch.tensor([self.get_word_index(w, training=False) for w in x], dtype=torch.long)
        probs = self.net(inputs)
        
        best_pos_ix = torch.argmax(probs).item()
        label = self.pos_from_ix[best_pos_ix]
        
        return label
        
    
    def test(self, XY):
        
        Ncorrect = 0
        Ntotal = len(XY)
        
        for x, y in XY:
            
            predicted_label = self.predict(x)
            
            if y == predicted_label:
                Ncorrect += 1
                
        return Ncorrect / Ntotal
        

class CustomFeatureNet(nn.Module):
    
    def __init__(self, V=30000, window_size=1, embedding_dim=50, num_labels=25):
        super(CustomFeatureNet, self).__init__()
        self.embedding = nn.Embedding(V, embedding_dim)
        self.linear1 = nn.Linear(3*embedding_dim + 3, 128)
        self.linear2 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.embedding.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds, features = inputs
        embeds = self.embedding(embeds).view((1, -1))
        out = torch.cat([embeds, features.view((1, -1))], dim=1)
        out = torch.tanh(self.linear1(out))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs
    
    
class CustomFeaturePOSTagger(BaselinePOSTagger):
    
    def __init__(self, word_to_ix={}, pos_to_ix={}, pos_from_ix={}, unk_ix=set(), window_size=1, lr=0.01):
        super(CustomFeaturePOSTagger, self).__init__(
            word_to_ix=word_to_ix, 
            pos_to_ix=pos_to_ix, 
            pos_from_ix=pos_from_ix, 
            unk_ix=unk_ix, 
            window_size=window_size,
            lr=lr,
            default_net=False
        )
        self.net = CustomFeatureNet(V=len(word_to_ix), window_size=window_size, num_labels=len(pos_to_ix))
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
    
    def get_features(self, x, training=True):
        
        embed_ixs = [self.get_word_index(w, training) for w in x]
        embed_ixs = torch.tensor(embed_ixs, dtype=torch.long)
        
        center = x[1]
        feature1 = math.log(len(center)) - 1
        feature2 = math.log(len(re.findall(r"[$%^&*()<>,./:;'\{\}_\-+=]", x[1])) + 1)
        feature3 = (int("!" in x[2]) - 0.5) / 10
        
        features = torch.tensor([feature1, feature2, feature3], dtype=torch.float)
        
        inputs = (embed_ixs, features)
        return inputs
        

class PreEmbedNet(nn.Module):
    
    def __init__(self, pre_embeds, window_size=1, num_labels=25, freeze=True):
        super(PreEmbedNet, self).__init__()
        embedding_dim = pre_embeds.size()[1]
        self.embedding = nn.Embedding.from_pretrained(pre_embeds, freeze=freeze)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1), 128)
        self.linear2 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        out = self.embedding(inputs).view((1, -1))
        out = torch.tanh(self.linear1(out))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs
    
class PreEmbedPOSTagger(BaselinePOSTagger):
    
    def __init__(self, pre_embeds, word_to_ix={}, pos_to_ix={}, pos_from_ix={}, unk_ix=set(), 
                 window_size=1, lr=0.01, freeze=True):
        super(PreEmbedPOSTagger, self).__init__(
            word_to_ix=word_to_ix, 
            pos_to_ix=pos_to_ix, 
            pos_from_ix=pos_from_ix, 
            unk_ix=unk_ix, 
            window_size=window_size,
            lr=lr,
            default_net=False
        )
        self.net = PreEmbedNet(pre_embeds, window_size=window_size, num_labels=len(pos_to_ix), freeze=freeze)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)


class CustomFeaturePreEmbedNet(nn.Module):
    
    def __init__(self, pre_embeds, window_size=1, num_labels=25, freeze=True):
        super(CustomFeaturePreEmbedNet, self).__init__()
        embedding_dim = pre_embeds.size()[1]
        self.embedding = nn.Embedding.from_pretrained(pre_embeds, freeze=freeze)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1) + 3, 128)
        self.linear2 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds, features = inputs
        embeds = self.embedding(embeds).view((1, -1))
        out = torch.cat([embeds, features.view((1, -1))], dim=1)
        out = torch.tanh(self.linear1(out))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs
    
class CustomPreEmPOSTagger(BaselinePOSTagger):
    
    def __init__(self, pre_embeds, word_to_ix={}, pos_to_ix={}, pos_from_ix={}, unk_ix=set(), 
                 window_size=1, lr=0.01, freeze=True):
        super(CustomPreEmPOSTagger, self).__init__(
            word_to_ix=word_to_ix, 
            pos_to_ix=pos_to_ix, 
            pos_from_ix=pos_from_ix, 
            unk_ix=unk_ix, 
            window_size=window_size,
            lr=lr,
            default_net=False
        )
        self.net = CustomFeaturePreEmbedNet(pre_embeds, window_size=window_size, 
                                            num_labels=len(pos_to_ix), freeze=freeze)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
    def get_features(self, x, training=True):
        
        embed_ixs = [self.get_word_index(w, training) for w in x]
        embed_ixs = torch.tensor(embed_ixs, dtype=torch.long)
        
        center = x[1]
        feature1 = math.log(len(center)) - 1
        feature2 = math.log(len(re.findall(r"[$%^&*()<>,./:;'\{\}_\-+=]", x[1])) + 1)
        feature3 = (int("!" in x[2]) - 0.5) / 10
        
        features = torch.tensor([feature1, feature2, feature3], dtype=torch.float)
        
        inputs = (embed_ixs, features)
        return inputs


class H2Net(nn.Module):
    
    def __init__(self, pre_embeds, window_size=1, num_labels=25, freeze=True):
        super(H2Net, self).__init__()
        embedding_dim = pre_embeds.size()[1]
        self.embedding = nn.Embedding.from_pretrained(pre_embeds, freeze=freeze)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1) + 3, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear3.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear3.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds, features = inputs
        embeds = self.embedding(embeds).view((1, -1))
        out = torch.cat([embeds, features.view((1, -1))], dim=1)
        out = torch.tanh(self.linear1(out))
        out = torch.tanh(self.linear2(out))
        out = self.linear3(out)
        probs = F.log_softmax(out, dim=1)
        return probs
    
class CustomArchPOSTagger(BaselinePOSTagger):
    
    def __init__(self, custom_net, pre_embeds, word_to_ix={}, pos_to_ix={}, pos_from_ix={}, unk_ix=set(), 
                 window_size=1, lr=0.01, freeze=True):
        super(CustomArchPOSTagger, self).__init__(
            word_to_ix=word_to_ix, 
            pos_to_ix=pos_to_ix, 
            pos_from_ix=pos_from_ix, 
            unk_ix=unk_ix, 
            window_size=window_size,
            lr=lr,
            default_net=False
        )
        self.net = custom_net(pre_embeds, window_size=window_size, 
                                            num_labels=len(pos_to_ix), freeze=freeze)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
    def get_features(self, x, training=True):
        
        embed_ixs = [self.get_word_index(w, training) for w in x]
        embed_ixs = torch.tensor(embed_ixs, dtype=torch.long)
        
        center = x[1]
        feature1 = math.log(len(center)) - 1
        feature2 = math.log(len(re.findall(r"[$%^&*()<>,./:;'\{\}_\-+=]", x[1])) + 1)
        feature3 = (int("!" in x[2]) - 0.5) / 10
        
        features = torch.tensor([feature1, feature2, feature3], dtype=torch.float)
        
        inputs = (embed_ixs, features)
        return inputs


class DoubleWideNet(nn.Module):
    
    def __init__(self, pre_embeds, window_size=1, num_labels=25, freeze=True):
        super(DoubleWideNet, self).__init__()
        embedding_dim = pre_embeds.size()[1]
        self.embedding = nn.Embedding.from_pretrained(pre_embeds, freeze=freeze)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1) + 3, 256)
        self.linear2 = nn.Linear(256, num_labels)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds, features = inputs
        embeds = self.embedding(embeds).view((1, -1))
        out = torch.cat([embeds, features.view((1, -1))], dim=1)
        out = torch.tanh(self.linear1(out))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs


class ReluNet(nn.Module):
    
    def __init__(self, pre_embeds, window_size=1, num_labels=25, freeze=True):
        super(ReluNet, self).__init__()
        embedding_dim = pre_embeds.size()[1]
        self.embedding = nn.Embedding.from_pretrained(pre_embeds, freeze=freeze)
        self.linear1 = nn.Linear(embedding_dim * (2 * window_size + 1) + 3, 128)
        self.linear2 = nn.Linear(128, num_labels)
        nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        
    def forward(self, inputs):
        embeds, features = inputs
        embeds = self.embedding(embeds).view((1, -1))
        out = torch.cat([embeds, features.view((1, -1))], dim=1)
        out = torch.relu(self.linear1(out))
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs
