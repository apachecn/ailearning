(function(){
    var footer = [
        '<hr/>', 
        '<div align="center">', 
        '  <p><a href="http://www.apachecn.org/" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a><p>', 
        '  <p><a href="https://github.com/apachecn/AiLearning/" target="_blank">apachecn/AiLearning</a></p>', 
        '  <p><iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=watch&count=true&v=2" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>', 
        '  <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=star&count=true" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>', 
        '  <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=fork&count=true" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>', 
        '  <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=bcee938030cc9e1552deb3bd9617bbbf62d3ec1647e4b60d9cd6b6e8f78ddc03"><img border="0" src="//pub.idqqimg.com/wpa/images/group.png" alt="ML | ApacheCN" title="ML | ApacheCN"></a></p>', 
        '  <p><span id="cnzz_stat_icon_1275211409"></span></p>', 
        '  <div style="text-align:center;margin:0 0 10.5px;">', 
        '    <ins class="adsbygoogle"', 
        '         style="display:inline-block;width:728px;height:90px"', 
        '         data-ad-client="ca-pub-3565452474788507"', 
        '         data-ad-slot="2543897000"></ins>', 
        '  </div>', 
        '</div>'
    ].join('\n')
    var plugin = function(hook) {
      hook.afterEach(function(html) {
        return html + footer
      })
      hook.doneEach(function() {
        (adsbygoogle = window.adsbygoogle || []).push({})
      })
    }
    var plugins = window.$docsify.plugins || []
    plugins.push(plugin)
    window.$docsify.plugins = plugins
})()