%%--Auto Encoder code by Narain Massand--%%
close all;
Loss=[];
loss=0;
CRtrain=[];
load('train.mat');
load('test.mat');
load('trainlabel.mat');
load('testlabel.mat');
load('label.mat');
alpha = 0.5;
eta = 0.1;
epochs = 50;
predict = zeros(1000,1);
I = 784;
H1 = 200;
O = 784;
b=[0.5 0.3];
wH = randn(I,H1);
wO = randn(H1,O);
wH = wH./sqrt(H1);
wO = wO./sqrt(O);
E = zeros(1,epochs);
wO_mom=zeros(H1,784);
wH_mom=zeros(784,H1);

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:4000
            r=k;
            
            % FORWARD PASS
            v1 = (train(r,:)*wH)+b(1);
            y1 = sigmoid(v1);
            v2 = (y1*wO)+b(2);
            y2 = sigmoid(v2);    
            
            % ERROR CALCULATION
            e = (train(r,:)-y2);
            cost=0.5*sum(e.^2);
            loss=loss+cost;
            
            % BACKWARD PASS
            changeO=((train(r,:)-y2)).*(sigmoid(v2).*(1-sigmoid(v2)));
            delta_wO=eta.*y1'*changeO+alpha.*wO_mom;
            delta_b2=eta.*changeO;
            wO=wO+delta_wO;
            b(2)=b(2)+(sum(delta_b2)/784);

            sumup=(wO*changeO')';
            changeH=(sigmoid(v1).*(1-sigmoid(v1))).*sumup;
            delta_wH=eta.*train(r,:)'*changeH+alpha.*wH_mom;
            delta_b1=eta.*changeH;
            wH=wH+delta_wH;
            b(1)=b(1)+(sum(delta_b1)/200);

            wO_mom=delta_wO;
            wH_mom=delta_wH;
    end
    
    % Error after each epoch
    Loss(end+1)=loss;
    loss=0;
end
toc;
figure;
plot(1:t,Loss,'b--o');
title('Loss Curve');
xlabel('Epochs');
ylabel('Loss');