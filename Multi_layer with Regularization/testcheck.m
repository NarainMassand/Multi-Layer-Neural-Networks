% % -- Implementing the trained network on TestData -- %%
    
for i=1:layer+1
    
    if (i==1)
        vT{i}=test*weight{i};
        yT{i}=sigmoid{vT{i}};
    else
        vT{i} = yT{i-1}*weight{i};
        yT{i} = sigmoid(vT{i});
    end
end
       
    % Predicting the values for each TestData input
    for z = 1:length(test)
        [val, col] = max(y2T(z,:));
        predicttest(z,:) = col-1;
    end
    % Calculating the Hit Rate in percentage
    CRt = (sum(predicttest==testlabel)/length(test))*100;
    ERt=100-CRt;