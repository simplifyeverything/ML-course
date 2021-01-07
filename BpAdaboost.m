%% ��ջ�������
clc
clear

%% ��������
load data input_train output_train input_test output_test

%% Ȩ�س�ʼ��
[inputnum,sampleNum]=size(input_train);
[outputnum,sampleNum]=size(output_train);
w(1,:)=ones(1,sampleNum)/sampleNum;%��ʼ������Ȩֵ����Ϊ1/sampleNum

%% ������������
iterate=10;%���û��������ĸ���
for i=1:iterate
    input_train=input_train.*w(i,:);%��Ȩѵ����
    %ѵ��������һ��
    [inputn,inputps]=mapminmax(input_train);
    [outputn,outputps]=mapminmax(output_train);
    error(i)=0;%��i�������������
    
    %BP�����繹��
    hiddennum=6;
    net=newff(inputn,outputn,hiddennum,{'tansig','tansig'},'trainlm');%tansigλ˫��sigmoid������trainlmΪLevenberg-Marquardt�㷨
    net.trainParam.epochs=10;%��������
    net.trainParam.lr=0.1;%ѧϰ����
    net.trainParam.goal=0.00004;%Ŀ�����
    net.divideFcn = ''; 
    %BP������ѵ��
    net=train(net,inputn,outputn);
    
    %ѵ������Ԥ��
    output1=sim(net,inputn);
    test_simu1(i,:)=mapminmax('reverse',output1,outputps);
    
    %��������Ԥ��
    inputn_test =mapminmax('apply',input_test,inputps);
    output=sim(net,inputn_test);
    test_simu(i,:)=mapminmax('reverse',output,outputps);
    
    %ͳ�����Ч��
    firstClassLocation=find(test_simu1(i,:)>0);%��һ���λ��
    secondClassLocation=find(test_simu1(i,:)<0);%�ڶ����λ��
    
    classifyResult(firstClassLocation)=1;
    classifyResult(secondClassLocation)=-1;
    
    %ͳ�ƴ���������
    for j=1:sampleNum
        if classifyResult(j)~=output_train(j)
            error(i)=error(i)+w(i,j);
        end
    end
    
    %��������iȨ��
    alfa(i)=0.5*log((1-error(i))/error(i));%��������Ȩ��
    
    %����wֵ
    for j=1:sampleNum
        w(i+1,j)=w(i,j)*exp(-alfa(i)*classifyResult(j)*test_simu1(i,j));
    end
    
    %wֵ��һ��
    wsum=sum(w(i+1,:));
    w(i+1,:)=w(i+1,:)/wsum;
    
end

%% ǿ������������
output=sign(alfa*test_simu);%��T����ѧϰ����Ȩ���

%% ������ͳ��
%ͳ��ǿ������ÿ�����������
firstClassifyError=0;
secondClassifyError=0;
for j=1:350
    if output(j)==1
        if output(j)~=output_test(j)
            firstClassifyError=firstClassifyError+1;
        end
    end
    if output(j)==-1
        if output(j)~=output_test(j)
            secondClassifyError=secondClassifyError+1;
        end
    end
end
fprintf("ͳ��ǿ����������Ч��\n");
fprintf("��һ��������=%d\n�ڶ���������=%d\n�ܴ���=%d\n",firstClassifyError,secondClassifyError,firstClassifyError+secondClassifyError);

plot(output,'*','DisplayName','Ԥ��ֵ')
hold on
plot(output_test,'*','DisplayName','��ʵֵ')
%ͳ����������Ч��
for i=1:iterate
    error1(i)=0;
    firstClassLocation=find(test_simu(i,:)>0);
    secondClassLocation=find(test_simu(i,:)<0);
    
    classifyResult(firstClassLocation)=1;
    classifyResult(secondClassLocation)=-1;
    
    for j=1:350
        if classifyResult(j)~=output_test(j)
            error1(i)=error1(i)+1;
        end
    end
end
fprintf("\nͳ��������������Ч��\n");
disp("ÿ�����������ķ��������");
disp(error1);
fprintf("ǿ���������������=%f\n�����������������=%f\n",(firstClassifyError+secondClassifyError)/350,(sum(error1)/(iterate*350)));

