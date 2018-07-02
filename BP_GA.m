input_train=input(1:1900,:)';
input_test=input(1901:2000,:)';
output_train=output(1:1900)';
output_test=output(1901:2000)';

inputnum=2;
hiddennum=5;
outputnum=1;

maxgen=50;         %最大迭代次数
sizepop=10;        %种群规模
pcross=0.4;        %交叉概率
pmutation=0.2;     %异变概率

%遗传算法主函数
function ga=genetic(inputnum,hiddennum,outputnum,input_train,output_train,maxgen,sizepop,pcross,pmutation)


%inputnum  输入节点数
%outputnum 输出节点数
%inputn    输入训练数
%outputn   输出训练数
%error     个体适应度

[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
%inputn为归一化数据，inputps为归一化数据体

net=newff(inputn,outputn,hiddennum);
%生成神经网络

sum_num=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
%节点总数

lenchrom=ones(1,sum_num);
%个体长度
bound=[-3*ones(1,sum_num) 3*ones(1,sum_num)];
%个体范围

individuals=struct('fitness',zeros(1,sizepop),'chrom',[]);
%个体信息数据
avgfitness=[];
%平均适应度
bestfitness=[];
%最佳适应度
bestchrom=[];
%最佳个体

for i=1:sizepop
    individuals.chrom(i,:)=code(lenchrom,bound);    %染色体编码，生成独立个体
  
    x=individuals.chrom(i,:);                       %利用神经网络计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
end
        
for i=1:maxgen
        individuals=select(individuals,sizepop);
        individuals.chrom=exchange(pcross,lenchrom,chrom,sizepop,bound);
        individuals.chrom=mutation(pmutation,lenchrom,chrom,sizepop,i,maxgen,bound);
        %从1至最大迭代数，进行选择、交换、变异
        for j=1:sizepop
            x=individuals.chrom(j,:);
            individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
        end
        %种群内计算基因算法作用后的适应度
        
        [newbestfitness,newbestindex]=min(individuals.fitness);
        [worstfitness,worstindex]=max(individuals.fitness);
        %计算新适应度
        
        if bestfitness>newbestfitness
            bestfitness=newbestfitness;
            bestchrom=individuals.chrom(newbestindex,:);
        end
        %判断是否为最优个体
        
        %更新适应度
        individuals.chrom(worstindex,:)=bestchrom;
        individuals.fitness(worstindex)=bestfitness;
        
        %计算平均适应度与个体适应度
        avgfitness=sum(individuals.fitness)/sizepop;
        trace=[trace;avgfitness bestfitness];
        
        %返回最优个体
        ga=individuals.chrom;
end


%利用神经网络计算适应度
function error=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)

w1=x(1:inputnum*hidennum);
b1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennnum*outputnum);
b2=x(inputnum*hiddennum+hiddennum+hiddennnum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennnum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddenum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,inputnum);
net.b{1}=reshape(b1,hiddennum,1);
net.b{2}=b2;

net.train.ep=20;%迭代次数
net.train.lr=0.1;%学习率
net.train.goal=0.00001;%目标
net.train.show=100;
net.train.showwin=0;

net=train(net,inputn,outputn);
an=sim(net,inputn);

error=sum(abs(an-putputn));


%染色体编码
function ret=code(lenchrom,bound)
flag=0;
while flag==0
    pick=rand(1,length(lenchrom));
    ret=bound(:,1)'+(bound(:,2)-bound(:,1))'.*pick; %线性插值，编码结果以实数向量存入ret中
    flag=test(lenchrom,bound,ret);     %检验染色体的可行性
end


%遗传算法选择操作
function ret=select(individuals,sizepop)

%计算适应度（倒数），适应度越小越好
fitness1=1./individuals.fitness;

%计算个体被选择概率,p=f/f(求和)
sumfitness=sum(fitness1);
pfitness=fitness1./sumfitness;

index=[];%选择序号索引

for i=1:sizepop
    pick=rand;
    while pick==0
        pick=rand;%未选择成功则重新选择
    end
    for j=1:sizepop
        pick=pick-pfitness(j);
        if pick<0
            index=[index j];%当适应度合适时，检索基因编号
            break;
        end
    end
end

%根据检索序号交换个体产生新种群
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;

%遗传算法交叉过程
function ret=exchange(pcross,lenchrom,chrom,sizepop,bound)

for i=1:sizepop
    
    %选择两个个体
    pick=rand(1,2);
    while prod(pick)==0
        pick=rand(1,2);
    end
    index=ceil(pick.*sizepop);%取整数选择个体序列号
    
    pick=rand;
    while pick==0
        pick=rand;
    end
    if pick>pcross%大于交叉概率则交叉
        continue;
    end
    flag=0;
    while flag==0
        pick=rand;
        while pick==0
            pick=rand;
        end
        pos=ceil(pick.*sum(lenchrom));%选择交叉位置
        
        %进行交叉操作，a=a1*(1-p)+a2*p
        pick=rand;
        v1=chrom(index(1),pos);
        v2=chrom(index(2),pos);
        chrom(index(1),pos)=pick*v2+(1-pick)*v1;
        chrom(index(2),pos)=pick*v1+(1-pick)*v2;
        
        %判断交叉结果是否满足要求
        flag1=test(lenchrom,bound,chrom(index(1),:));
        flag2=test(lenchrom,bound,chrom(index(2),:));
        
        %验证两个染色体是否均可行（不为0），若不为零则运算完成
        if flag1*flag2==0
            flag=0;
        else
            flag=1;
        end
    end
end

ret=chrom;

%遗传算法变异操作
function ret=mutation(pmutation,lenchrom,chrom,sizepop,num,maxgen,bound)

for i=1:sizepop
    pick=rand;
    while pick==0
        pick=rand;
    end
    index=ceil(pick*sizepop);
    
    pick=rand;
    if pick>pmutation
        continue;
    end
    flag=0;
    while flag==0
        pick=rand;
        while pick==0
            pick=rand;
        end
        pos=ceil(pick*sum(lenchrom));%随机选择变异位置
        
        %变异计算
        pick=rand;
        fg=(rand*(1-num/maxgen))^2;
        if pick>0.5
            chrom(i.pos)=chrom(i,pos)+(chrom(i,pos)-bound(i,pos))*fg;
        else
            chrom(i.pos)=chrom(i,pos)+(bound(i,pos)-chrom(i,pos))*fg;
        end
        flag=test(lenchrom,bound,chrom(i,:));
    end
end
ret=chrom;





        





        
        
