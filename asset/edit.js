document.addEventListener('DOMContentLoaded', function() {
	var editBtn = document.createElement('div')
	editBtn.id = 'edit-btn'
	document.body.append(editBtn)
	
	var repo = window.$docsify.repo
	editBtn.addEventListener('click', function() {
		if (!repo) return
		if (!/https?:\/\//.exec(repo))
			repo = 'https://github.com/' + repo
		var url = repo + '/tree/master' + 
			      location.hash.slice(1) + '.md'
		window.open(url)
	})
})