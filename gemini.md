1. 查看我的main函数、SR-D.py函数以及SR-D-MATLAB文件夹下的内容

- 注意到，我的SR-D函数是根据SR-D-MATLAB内容修改的，一切以SR-D-MATLAB为准

2. 我目前的SR-D.py函数除了在边缘处理上和SR-D-MATLAB的处理方式不同之外，其他方式我想都是一致的

- 目前PAN的大小是256x255, MS是64x64, **在这种情况下 pad = 0, 边缘不需要处理，即两种方式应当是完全等效的**

3. 目前我已经做过的实验及其结果记录在了`Record.md`文件中的`SR-D测试`部分，请你**仔细看结果**

- Almost all the PSNR results are 30.29dB, Which is the same as LMS!!!!

- Please find any python codes that implement the different functionalities compared with original MATLAB version

- 你给出的解决方案的**最终结果应当和MATLAB接近甚至相同**

4. 任何可能的代码修改都必须符合我源代码的风格，不要擅自加一些我不喜欢的风格!!!!!!!!!!!!!!!!!!!!!!!

5. **每一次的实验结果都要详细记录**到`Record.md`文件的`SR-D测试`部分的表格里

- **请不要尝试已经在实验结果中记录过的方法**

6. Always, always answer in Chinese!!!

7. Scroll back to the original version if it all fails!!
