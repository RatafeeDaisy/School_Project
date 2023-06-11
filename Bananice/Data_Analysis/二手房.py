zone = sns.FacetGrid(data,col='区域',col_wrap=2)
zone.map(sns.jointplot(x="价格（W）", y="单价（平方米）",kind="reg"))