 % Testing the TestData using the updated weight matrices
    v1T = test*wHA;
    y1T = sigmoid(v1T);
  
    v2T = y1T*wO;
    y2T = sigmoid(v2T);
    
    for z = 1:length(test)
        [val, col] = max(y2T(z,:));
        predicttest(z,:) = col-1;
    end
        
    % Calculating the Hit Rate in percentage
    CRt = (sum(predicttest==testlabel)/length(test))*100;
    ERt=100-CRt;