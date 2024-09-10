// 定义一个名为scrollToPage的函数，用于滚动到指定页码的页面元素
function scrollToPage(page) {
    // 根据页码获取对应的页面元素
    var elem = document.getElementById("page" + page);
    // 如果找到了页面元素
    if (elem) {
        // 使用平滑滚动将页面元素滚动到可视区域中心位置
        elem.scrollIntoView({behavior: "smooth", block: "center"});
    }
}

// 查找所有以"#page"开头的锚点元素，并为它们添加点击事件监听器
document.querySelectorAll('a[href^="#page"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        // 阻止默认点击事件的发生
        e.preventDefault();
        // 获取锚点的目标页码
        var pageId = this.getAttribute('href').substring(1);
        // 从锚点的目标页码中提取页码数字
        var page = pageId.replace('page', '');
        // 调用scrollToPage函数，将页面滚动到相应的页码
        scrollToPage(page);
    });
});