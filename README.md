# 解决：Uncaught TypeError: Cannot read property 'list' of undefined

## 问题描述
在 Vue3 项目中，接口请求后赋值时报错：
`Cannot read property 'list' of undefined`

## 代码片段
```javascript
async getList() {
  const res = await api.fetchData();
  // 报错行
  this.tableData = res.data.list;
}
