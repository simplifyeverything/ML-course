%% 清空环境变量
clc
clear

%% 加载数据
load data input_train output_train input_test output_test

%% 权重初始化
[inputnum,sampleNum]=size(input_train);
[outputnum,sampleNum]=size(output_train);
w(1,:)=ones(1,sampleNum)/sampleNum;%初始化样本权值，都为1/sampleNum

%% 弱分类器分类
iterate=10;%设置基分类器的个数
for i=1:iterate
    input_train=input_train.*w(i,:);%带权训练集
    %训练样本归一化
    [inputn,inputps]=mapminmax(input_train);
    [outputn,outputps]=mapminmax(output_train);
    error(i)=0;%第i个分类器的误差
    
    %BP神经网络构建
    hiddennum=6;
    net=newff(inputn,outputn,hiddennum,{'tansig','tansig'},'trainlm');%tansig位双曲sigmoid函数，trainlm为Levenberg-Marquardt算法
    net.trainParam.epochs=10;%迭代次数
    net.trainParam.lr=0.1;%学习速率
    net.trainParam.goal=0.00004;%目标误差
    net.divideFcn = ''; 
    %BP神经网络训练
    net=train(net,inputn,outputn);
    
    %训练数据预测
    output1=sim(net,inputn);
    test_simu1(i,:)=mapminmax('reverse',output1,outputps);
    
    %测试数据预测
    inputn_test =mapminmax('apply',input_test,inputps);
    output=sim(net,inputn_test);
    test_simu(i,:)=mapminmax('reverse',output,outputps);
    
    %统计输出效果
    firstClassLocation=find(test_simu1(i,:)>0);%第一类的位置
    secondClassLocation=find(test_simu1(i,:)<0);%第二类的位置
    
    classifyResult(firstClassLocation)=1;
    classifyResult(secondClassLocation)=-1;
    
    %统计错误样本数
    for j=1:sampleNum
        if classifyResult(j)~=output_train(j)
            error(i)=error(i)+w(i,j);
        end
    end
    
    %弱分类器i权重
    alfa(i)=0.5*log((1-error(i))/error(i));%分类器的权重
    
    %更新w值
    for j=1:sampleNum
        w(i+1,j)=w(i,j)*exp(-alfa(i)*classifyResult(j)*test_simu1(i,j));
    end
    
    %w值归一化
    wsum=sum(w(i+1,:));
    w(i+1,:)=w(i+1,:)/wsum;
    
end

%% 强分类器分类结果
output=sign(alfa*test_simu);%将T个基学习器加权结合

%% 分类结果统计
%统计强分类器每类分类错误个数
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
fprintf("统计强分类器分类效果\n");
fprintf("第一类分类错误=%d\n第二类分类错误=%d\n总错误=%d\n",firstClassifyError,secondClassifyError,firstClassifyError+secondClassifyError);

plot(output,'*','DisplayName','预测值')
hold on
plot(output_test,'*','DisplayName','真实值')
%统计弱分类器效果
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
fprintf("\n统计弱分类器分类效果\n");
disp("每次弱分类器的分类错误数");
disp(error1);
fprintf("强分类器分类误差率=%f\n弱分类器分类误差率=%f\n",(firstClassifyError+secondClassifyError)/350,(sum(error1)/(iterate*350)));

