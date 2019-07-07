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
eta = 0.001;
epochs = 100;
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

initial_activation=[]; %activation matrix
ro(1:200,1)=0.05; %desired average activation
ro_actual=[];
actvn_div=[];
beta=4;

lambda=0.0005;

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    
    initial_activation=[];
    for k=1:4000
        v = (train(k,:)*wH)+b(1);
        y = sigmoid(v);
        initial_activation(:,end+1)=y;
    end
    
    ro_actual=(sum(initial_activation,2))/4000;
    actvn_div=(ro.*log2(ro./ro_actual))+((1-ro).*log2((1-ro)./(1-ro_actual)));
    
    for k = 1:500
%             r=k;
            
%  Random dataset row selection parameter
            r = round(length(train).*rand(1,1));
            if r == 0
                r = r+1;
            end
            
            % FORWARD PASS
            v1 = (train(r,:)*wH)+b(1);
            y1 = sigmoid(v1);
            v2 = (y1*wO)+b(2);
            y2 = sigmoid(v2);    
            
            % ERROR CALCULATION
            e = (train(r,:)-y2);
            cost=0.5*sum(e.^2)+((lambda/2)*(sum(sum(wH.^2))+sum(sum(wO.^2))))+(beta*sum(actvn_div));
%             cost=0.5*sum(e.^2)+(beta*sum(actvn_div));
            loss=loss+cost;
            
            % BACKWARD PASS
            changeO=((train(r,:)-y2)).*(sigmoid(v2).*(1-sigmoid(v2)));
            delta_wO=eta.*((y1'*changeO)-(lambda*wO))+alpha.*wO_mom;
            delta_b2=eta.*changeO;
            wO=wO+delta_wO;
            b(2)=b(2)+(sum(delta_b2)/784);

            sumup=(wO*changeO')';
%             newsumup=(sumup-(beta*(((1-ro)./(1-ro_actual))-(ro./ro_actual))))';
            changeH=((sigmoid(v1).*(1-sigmoid(v1))).*sumup);
%             -(beta*(((1-ro)./(1-ro_actual))-(ro./ro_actual)))';
            delta_wH=eta.*((train(r,:)'*changeH)-(lambda*wH))+alpha.*wH_mom;
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