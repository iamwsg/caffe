function r=smax(r1)

r=exp(r1);
sr=sum(r);

r=r./sr;
end