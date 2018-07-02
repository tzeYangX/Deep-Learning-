input_train=input(1:1900,:)';
input_test=input(1901:2000,:)';
output_train=output(1:1900)';
output_test=output(1901:2000)';

inputnum=2;
hiddennum=5;
outputnum=1;

maxgen=50;         %����������
sizepop=10;        %��Ⱥ��ģ
pcross=0.4;        %�������
pmutation=0.2;     %������

%�Ŵ��㷨������
function ga=genetic(inputnum,hiddennum,outputnum,input_train,output_train,maxgen,sizepop,pcross,pmutation)


%inputnum  ����ڵ���
%outputnum ����ڵ���
%inputn    ����ѵ����
%outputn   ���ѵ����
%error     ������Ӧ��

[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
%inputnΪ��һ�����ݣ�inputpsΪ��һ��������

net=newff(inputn,outputn,hiddennum);
%����������

sum_num=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
%�ڵ�����

lenchrom=ones(1,sum_num);
%���峤��
bound=[-3*ones(1,sum_num) 3*ones(1,sum_num)];
%���巶Χ

individuals=struct('fitness',zeros(1,sizepop),'chrom',[]);
%������Ϣ����
avgfitness=[];
%ƽ����Ӧ��
bestfitness=[];
%�����Ӧ��
bestchrom=[];
%��Ѹ���

for i=1:sizepop
    individuals.chrom(i,:)=code(lenchrom,bound);    %Ⱦɫ����룬���ɶ�������
  
    x=individuals.chrom(i,:);                       %���������������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
end
        
for i=1:maxgen
        individuals=select(individuals,sizepop);
        individuals.chrom=exchange(pcross,lenchrom,chrom,sizepop,bound);
        individuals.chrom=mutation(pmutation,lenchrom,chrom,sizepop,i,maxgen,bound);
        %��1����������������ѡ�񡢽���������
        for j=1:sizepop
            x=individuals.chrom(j,:);
            individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
        end
        %��Ⱥ�ڼ�������㷨���ú����Ӧ��
        
        [newbestfitness,newbestindex]=min(individuals.fitness);
        [worstfitness,worstindex]=max(individuals.fitness);
        %��������Ӧ��
        
        if bestfitness>newbestfitness
            bestfitness=newbestfitness;
            bestchrom=individuals.chrom(newbestindex,:);
        end
        %�ж��Ƿ�Ϊ���Ÿ���
        
        %������Ӧ��
        individuals.chrom(worstindex,:)=bestchrom;
        individuals.fitness(worstindex)=bestfitness;
        
        %����ƽ����Ӧ���������Ӧ��
        avgfitness=sum(individuals.fitness)/sizepop;
        trace=[trace;avgfitness bestfitness];
        
        %�������Ÿ���
        ga=individuals.chrom;
end


%���������������Ӧ��
function error=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)

w1=x(1:inputnum*hidennum);
b1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennnum*outputnum);
b2=x(inputnum*hiddennum+hiddennum+hiddennnum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennnum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddenum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,inputnum);
net.b{1}=reshape(b1,hiddennum,1);
net.b{2}=b2;

net.train.ep=20;%��������
net.train.lr=0.1;%ѧϰ��
net.train.goal=0.00001;%Ŀ��
net.train.show=100;
net.train.showwin=0;

net=train(net,inputn,outputn);
an=sim(net,inputn);

error=sum(abs(an-putputn));


%Ⱦɫ�����
function ret=code(lenchrom,bound)
flag=0;
while flag==0
    pick=rand(1,length(lenchrom));
    ret=bound(:,1)'+(bound(:,2)-bound(:,1))'.*pick; %���Բ�ֵ����������ʵ����������ret��
    flag=test(lenchrom,bound,ret);     %����Ⱦɫ��Ŀ�����
end


%�Ŵ��㷨ѡ�����
function ret=select(individuals,sizepop)

%������Ӧ�ȣ�����������Ӧ��ԽСԽ��
fitness1=1./individuals.fitness;

%������屻ѡ�����,p=f/f(���)
sumfitness=sum(fitness1);
pfitness=fitness1./sumfitness;

index=[];%ѡ���������

for i=1:sizepop
    pick=rand;
    while pick==0
        pick=rand;%δѡ��ɹ�������ѡ��
    end
    for j=1:sizepop
        pick=pick-pfitness(j);
        if pick<0
            index=[index j];%����Ӧ�Ⱥ���ʱ������������
            break;
        end
    end
end

%���ݼ�����Ž��������������Ⱥ
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;

%�Ŵ��㷨�������
function ret=exchange(pcross,lenchrom,chrom,sizepop,bound)

for i=1:sizepop
    
    %ѡ����������
    pick=rand(1,2);
    while prod(pick)==0
        pick=rand(1,2);
    end
    index=ceil(pick.*sizepop);%ȡ����ѡ��������к�
    
    pick=rand;
    while pick==0
        pick=rand;
    end
    if pick>pcross%���ڽ�������򽻲�
        continue;
    end
    flag=0;
    while flag==0
        pick=rand;
        while pick==0
            pick=rand;
        end
        pos=ceil(pick.*sum(lenchrom));%ѡ�񽻲�λ��
        
        %���н��������a=a1*(1-p)+a2*p
        pick=rand;
        v1=chrom(index(1),pos);
        v2=chrom(index(2),pos);
        chrom(index(1),pos)=pick*v2+(1-pick)*v1;
        chrom(index(2),pos)=pick*v1+(1-pick)*v2;
        
        %�жϽ������Ƿ�����Ҫ��
        flag1=test(lenchrom,bound,chrom(index(1),:));
        flag2=test(lenchrom,bound,chrom(index(2),:));
        
        %��֤����Ⱦɫ���Ƿ�����У���Ϊ0��������Ϊ�����������
        if flag1*flag2==0
            flag=0;
        else
            flag=1;
        end
    end
end

ret=chrom;

%�Ŵ��㷨�������
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
        pos=ceil(pick*sum(lenchrom));%���ѡ�����λ��
        
        %�������
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





        





        
        
