%%--Auto Encoder with regularization by Narain Massand--%%
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
eta = 0.01;
epochs = 800;
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
delta_wHacc=zeros(784,200);
delta_b1acc=zeros(1,200);
delta_wOacc=zeros(200,784);
delta_b2acc=zeros(1,784);
initial_activation=[]; %activation matrix
ro(1:200,1)=0.05; %desired average activation
ro_actual=[];
actvn_div=[];
beta=2;

lambda=0.001;

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    %calculating initial activation
    initial_activation=[];
    
%     if (t>=300 && t<=800)
%         eta=0.002;
%     elseif (t>=600)
%         eta=0.001;
%     end
    
    for k=1:4000
        v = (train(k,:)*wH)+b(1);
        y = sigmoid(v);
        initial_activation(:,end+1)=y;
    end
    
    ro_actual=(sum(initial_activation,2))/4000;
    actvn_div=(ro.*log10(ro./ro_actual))+((1-ro).*log10((1-ro)./(1-ro_actual)));
     
    for k = 1:500
%             r=k;
             r = round(length(train).*rand(1,1));
            if r == 0
                r = r+1;
            end
            
            % FORWARD PASS
            v1 = (train(r,:)*wH)+b(1);
            y1 = sigmoid(v1);
            v2 = (y1*wO)+b(2);
            y2 = sigmoid(v2);    
           
%             %Calculating activations and divergence
%             initial_activation(:,end+1)=y1;
%             ro_actual=(sum(initial_activation,2))/500;
%             actvn_div=(ro.*log10(ro./ro_actual))+((1-ro).*log10((1-ro)./(1-ro_actual)));
%     
            
            % ERROR CALCULATION
            e = (train(r,:)-y2);
            cost=0.5*sum(e.^2);
%             (beta*sum(actvn_div))+((lambda/2)*(sum(sum(wH.^2))+sum(sum(wO.^2))));
            loss=loss+cost;
            
            % BACKWARD PASS
            changeO=((train(r,:)-y2)).*(sigmoid(v2).*(1-sigmoid(v2)));
            delta_wO=eta.*((y1'*changeO)-(lambda*wO))+alpha.*wO_mom;
%             delta_wO=eta.*y1'*changeO+alpha.*wO_mom;
            delta_wOacc=delta_wOacc+delta_wO;
            delta_b2=eta.*changeO;
            delta_b2acc=delta_b2acc+delta_b2;
%             wO=wO+delta_wO;
%             b(2)=b(2)+mean(delta_b2);

            sumup=((wO*changeO')-(beta.*(((1-ro)./(1-ro_actual))-(ro./ro_actual))))';
            changeH=(sigmoid(v1).*(1-sigmoid(v1))).*sumup;
            delta_wH=eta.*((train(r,:)'*changeH)-(lambda*wH))+alpha.*wH_mom;
%             delta_wH=eta.*train(r,:)'*changeH+alpha.*wH_mom;
            delta_wHacc=delta_wHacc+delta_wH;
            delta_b1=eta.*changeH;
            delta_b1acc=delta_b1acc+delta_b1;
%             wH=wH+delta_wH;
%             b(1)=b(1)+mean(delta_b1);

%             wO_mom=eta.*y1'*changeO+alpha.*wO_mom;
%             wH_mom=eta.*train(r,:)'*changeH+alpha.*wH_mom;
            
            wO_mom=delta_wO;
            wH_mom=delta_wH;
    end
    %Updating weights and biases after a whole batch
    wO=wO+delta_wOacc;
    b(2)=b(2)+mean(delta_b2acc);
    wH=wH+delta_wHacc;
    b(1)=b(1)+mean(delta_b1acc);
    delta_wHacc=0;
    delta_wOacc=0;
    delta_b1acc=0;
    delta_b2acc=0;
     % Error after each epoch
    Loss(end+1)=loss;
    loss=0;
end
toc;
figure;
plot(1:t,Loss,'b--o');
title('Fig 1.3 Time series of Error');
xlabel('Epochs');
ylabel('Error');