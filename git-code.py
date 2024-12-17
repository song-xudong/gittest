echo "# gittest" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/song-xudong/gittest.git
git push -u origin main


git remote add origin https://github.com/song-xudong/gittest.git
git branch -M main
git push -u origin main

#将所有内容提交
git add . 

#新增快照，名字为"first commit"
git commit -m "first commit"

#提交到github

git push