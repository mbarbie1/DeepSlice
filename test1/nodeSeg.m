%% Brain image:
BI=zeros(300);

[X Y]=meshgrid([-150:150],[-150:150]);
BI=((sqrt(X.^2+Y.^2))<40);
imagesc(BI)

%% Nodes
% create nodes
t=1:36:360;

nodes= round(150+140*[cos(t/360*2*pi); sin(t/360*2*pi)]);
%%
figure; 
%%
plot(nodes(1,:),nodes(2,:),'*')
hold on;
plot(nodes(1,:),nodes(2,:))
hold off

%%
ln=length(nodes);
for i=1:length(nodes)
    gotoNode=nodes(1,i)-.1*(nodes(1,i)-150);
    nn=mod(i+1,ln)+1; %Next node
    if nn<ln/2 % upper circle
    order=(nodes(1,mod(i+1,ln)+1)<gotoNode); % Check order in X
    end
    if nn>ln/2 % lower circle
    order=(nodes(1,mod(i+1,ln)+1)>gotoNode);
    end
    
    
    if BI(round(gotoNode),round(nodes(2,i)))<1 && order
        nodes(1,i)=gotoNode;
    end
    gotoNode=nodes(2,i)-.1*(nodes(2,i)-150);
    if (BI(round(nodes(1,i)),round(gotoNode))<1)
        nodes(2,i)=gotoNode;
    end    
end

imagesc(BI)
hold on;
plot(nodes(1,:),nodes(2,:),'*')
hold on;
plot(nodes(1,:),nodes(2,:))
hold off
