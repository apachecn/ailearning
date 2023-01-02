(function(){
	var cnzzId = window.$docsify.cnzzId
	var unRepo = window.$docsify.repo || ''
	var [un, repo] = unRepo.split('/')
    var footer = `
        <hr/>
        <div align="center">
          <p><a href="http://www.apachecn.org/" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a><p>
          <p><a href="https://github.com/${unRepo}" target="_blank">${unRepo}</a></p>
          <p><a target="_blank" href="https://qm.qq.com/cgi-bin/qm/qr?k=2oFEp1KHbDCP0Te4Wt-I6FOK4hvg4iBk&jump_from=webapi&authKey=dw08LmD1w9km55TSmcW2J4gjeaiyn7KTff+8bnqiIeDweqrzQF2ccsE/hQswWQk7"><img border="0" src="//pub.idqqimg.com/wpa/images/group.png" alt="iBooker 面试求职" title="iBooker 面试求职"></a></p>
          <p><span id="cnzz_stat_icon_${cnzzId}"></span></p>
          <div class="wwads-cn wwads-horizontal" data-id="206" style="max-width:350px"></div>
          <div style="text-align:center;margin:0 0 10.5px;">
            <ins class="adsbygoogle"
                 style="display:inline-block;width:728px;height:90px"
                 data-ad-client="ca-pub-3565452474788507"
                 data-ad-slot="2543897000"></ins>
          </div>
        </div>
	`
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