%%--Back-propagation code by Narain Massand--%%
close all;
load('train.mat');
load('test.mat');
load('trainlabel.mat');
load('testlabel.mat');
load('label.mat');
alpha = 1;
eta = 0.09;
epochs = 20;
predict = zeros(1000,1);
I = 784;
% H1 = ('Enter the number of Hidden layer 1 units: ');
% H1 =  input(H1);
% H2 = ('Enter the number of Hidden layer 2 units: ');
% H2 =  input(H2);
% O = ('Enter the number of Output layer units: ');
% O =  input(O);
H1 = 200;
% H2 = 50;
O = 10;
% b=[0.5 0.3];
wH = randn(I,H1);
% wH2 = randn(H1,H2);
wO = randn(H1,O);
wH = wH./sqrt(H1);
% wH2 = wH2./sqrt(H2);
wO = wO./sqrt(O);
E = zeros(1,epochs);

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:length(train)
            % Random dataset row selection parameter
            r = round(length(train).*rand(1,1));
            if r == 0
                r = r+1;
            end
            % FORWARD PASS
            v1 = (train(r,:)*wH)+b(1);
            y1 = sigmoid(v1);
            v2 = (y1*wO)+b(2);
            y2 = sigmoid(v2);    
%             v3 = (y2*wO)+b(3);
%             y3 = sigmoid(v3);
            
            % ERROR CALCULATION
            e = label(r,:)-y2;
            cost=0.5*sum(e.^2);
            changeO=(label(r,:)-y2).*(sigmoid(v2).*(1-sigmoid(v2)));
            delta_wO=eta.*y1'*changeO;
            delta_b2=eta.*changeO;
            wO=wO+delta_wO;
            b(2)=b(2)+(sum(delta_b2)/10);

            sumup=(wO*changeO')';
            changeH=(sigmoid(v1).*(1-sigmoid(v1))).*sumup;
            delta_wH=eta.*train'*changeH;
            delta_b1=eta.*changeH;
            wH=wH+delta_wH;
            b(1)=b(1)+(sum(delta_b1)/200);

%             del=((label(r,:)-y2).*(sigmoid(v2).*(1-sigmoid(v2)))).*wO;
%             changeH=(sigmoid(v1).*(1-sigmoid(v1)))*del;
%              b(2)=b(2)+eta*changeO;
%             wH=wH+eta*changeH;
%             b(1)=b(1)+eta*changeH;
            
            
%             % BACKWARD PASS
%             dO  = dlogsig(v2,y2).*(e);
% %             dbO  = dlogsig(v3,y3).*(e);
% %             dH2 = diag(dlogsig(v2,y2))*(wO*dO');
%             dH1  = diag(dlogsig(v1,y1))*(wO*dO);
%             
%             
%             wO  = alpha.*wO +(eta.*dO'*y2)';
% %             b(3)=
% %             wH2 = alpha.*wH2 +(eta.*dH2*y1)';
%             wH  = alpha.*wH +(eta.*dH1*train(r,:))';
    end
    
    % Error after each epoch
%     E(t) = sum(e.^2)/10;
E(t) = sum(cost)/10;
    % Testing the TestData using the updated weight matrices
    v1T = test*wH;
    y1T = sigmoid(v1T);
%     v2T = y1T*wH2;
%     y2T = logsig(v2T);    
    v2T = y1T*wO;
    y2T = sigmoid(v2T);
    
    % Predicting the values for each TestData input
    for z = 1:length(test)
        [val, col] = max(y2T(z,:));
        predict(z,:) = col-1;
    end
    % Calculating the Classification Rate in percentage
    CR(t) = (sum(predict==testlabel)/length(test))*100;
end
toc;
