%%--HW 4, problem 2, Network A by Narain Massand--%%
close all;
CRtrain=[];
load('train.mat');
load('test.mat');
load('trainlabel.mat');
load('testlabel.mat');
load('label.mat');
alpha = 0.5;
eta = 0.1;
epochs = 400;
predict = zeros(1000,1);
I = 784;
A=zeros(1,10);
B=zeros(1,200);

H1 = 200;
O = 10;
b=[0.5 0.3];
% wH = randn(I,H1);
wHA=netBwH;
% wHA=wHCopy;
wO = randn(H1,O);
wO = wO./sqrt(O);
E = zeros(1,epochs);
wO_mom=zeros(H1,10);
loss=0;
Loss=[];

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:4000
r=k;
        % Random dataset row selection parameter
%             r = round(length(train).*rand(1,1));
%             if r == 0
%                 r = r+1;
%             end
            % FORWARD PASS
            v1 = (train(r,:)*wHA)+b(1);
            y1 = sigmoid(v1);
            v2 = (y1*wO)+b(2);
            y2 = sigmoid(v2);    
%           v3 = (y2*wO)+b(3);
%           y3 = sigmoid(v3);
            
            % ERROR CALCULATION
            e = label(r,:)-y2;
            cost=0.5*sum(e.^2);
            loss=loss+cost;
            % BACKWARD PASS
            
            changeO=(label(r,:)-y2).*(sigmoid(v2).*(1-sigmoid(v2)));
            delta_wO=eta.*y1'*changeO+alpha.*wO_mom;
            delta_b2=eta.*changeO;
            wO=wO+delta_wO;
            b(2)=b(2)+(sum(delta_b2)/10);

%             sumup=(wO*changeO')';
%             changeH=(sigmoid(v1).*(1-sigmoid(v1))).*sumup;
%             delta_wH=eta.*train(r,:)'*changeH+alpha.*wH_mom;
%             delta_b1=eta.*changeH;
%             wH=wH+delta_wH;
%             b(1)=b(1)+(sum(delta_b1)/200);

            wO_mom=delta_wO;
%             wH_mom=delta_wH;
%     
%             del=((label(r,:)-y2).*(sigmoid(v2).*(1-sigmoid(v2)))).*wO;
%             changeH=(sigmoid(v1).*(1-sigmoid(v1)))*del;
%              b(2)=b(2)+eta*changeO;
%             wH=wH+eta*changeH;
%             b(1)=b(1)+eta*changeH;
            
            

    end
    
    %
%     wO_mom=[A;delta_wO];
%     wH_mom=[B;delta_wH];
%     wO_mom(end,:)=[];
%     wH_mom(end,:)=[];
    % Error after each epoch
    Loss(t)=loss;
    loss=0;
%     E(t) = sum(e.^2)/10;
% E(t) = sum(cost)/10;
%     % Testing the TestData using the updated weight matrices
%     v1T = test*wH;
%     y1T = sigmoid(v1T);
% %     v2T = y1T*wH2;
% %     y2T = logsig(v2T);    
%     v2T = y1T*wO;
%     y2T = sigmoid(v2T);
%     
%     % Predicting the values for each TestData input
%     for z = 1:length(test)
%         [val, col] = max(y2T(z,:));
%         predict(z,:) = col-1;
%     end
%     % Calculating the Classification Rate in percentage
%     CR(t) = (sum(predict==testlabel)/length(test))*100;

%%calculating the hit rate on training set per epoch

v1Tr = train*wHA;
    y1Tr = sigmoid(v1Tr);
%     v2T = y1T*wH2;
%     y2T = logsig(v2T);    
    v2Tr = y1Tr*wO;
    y2Tr = sigmoid(v2Tr);
    
        for z = 1:length(train)
        [val, col] = max(y2Tr(z,:));
        predict(z,:) = col-1;
    end
    % Calculating the Classification Rate in percentage
    CRtrain(t) =(sum(predict==trainlabel)/length(train))*100;
end
toc;
figure;
plot(1:t,CRtrain,'b--o');
title('Recognition Curve');
xlabel('Epochs');
ylabel('Classification Rate (%)');
figure;
plot(1:t,Loss,'b--o')
title('Loss Curve');
xlabel('Epochs');
ylabel('Loss');
errorrate=[];
for i=1:10:400
errorrate(end+1)=100-CRtrain(i);
end
figure;
plot(1:10:t,errorrate,'b--o');
title('Time series of Error');
xlabel('Epochs');
ylabel('Error Rate (%)');
title('Fig 2.2 Time series of Error - Network B');
ylabel('Error Rate (%)');