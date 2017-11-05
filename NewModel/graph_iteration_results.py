import matplotlib.pyplot as plt

nb_result_list = [(0.68398392652123996, 0.68398392652123996), (0.77037887485648682, 0.77037887485648682), (0.81199770378874858, 0.81199770378874858), (0.098163030998851888, 0.098163030998851902), (0.74196326061997708, 0.74196326061997708)]

rf_result_list = [(0.6676234213547646, 0.6676234213547646), (0.61567164179104472, 0.61567164179104472), (0.63002296211251441, 0.63002296211251441), (0.62313432835820892, 0.62313432835820892), (0.5875430539609644, 0.5875430539609644)]
svm_result_list = [(0.81659012629161887, 0.81659012629161876), (0.78099885189437424, 0.78099885189437412), (0.66819747416762343, 0.66819747416762343), (0.50545350172215842, 0.50545350172215842), (0.36452353616532723, 0.36452353616532718)]

eval_score_confidence_list = [(3.8375479769375227, 2.0564700036530259, 5.6186259502220199), (-0.44626168663475063, -1.6820720262342541, 0.78954865296475285), (0.82960427166892647, -0.56041867302545922, 2.2196272163633122), (-0.28905312889451629, -2.0035716303603355, 1.4254653725713027), (1.1264634257528194, -0.55890334182998647, 2.8118301933356253)]

p_score_confidence_list = [(21.058171010014387, -1.4088593144943715, 43.525201334523146), (53.410261888277475, 30.742083646359767, 76.07844013019519), (55.081604683323881, 30.738231814111845, 79.424977552535921), (34.084944040733326, 20.028291556739475, 48.141596524727177), (38.449744877101409, 15.43870531099531, 61.460784443207508)]

#plot eval score confidence interval over the rounds

eval_score_means = [a[0] for a in eval_score_confidence_list]
eval_score_lowers =  [a[1] for a in eval_score_confidence_list]
eval_score_uppers = [a[2] for a in eval_score_confidence_list]
p_score_means = [a[0]*.1 for a in p_score_confidence_list]
p_score_lowers =  [a[1]*.1 for a in p_score_confidence_list]
p_score_uppers = [a[2]*.1 for a in p_score_confidence_list]

nb_results_scaled = map(lambda x: x[0]*10,nb_result_list)
rf_results_scaled = map(lambda x: x[0]*10,rf_result_list)
svm_results_scaled = map(lambda x: x[0]*10,svm_result_list)

round_list = xrange(0,len(eval_score_means))
plt.plot(round_list,eval_score_means)
plt.fill_between(round_list,eval_score_lowers,eval_score_uppers,alpha=0.2)

plt.plot(round_list,p_score_means,color='m')
plt.fill_between(round_list,p_score_lowers,p_score_uppers,color='m',alpha=0.2)

plt.plot(round_list,nb_results_scaled, color ='r')
plt.plot(round_list,rf_results_scaled, color ='y')
plt.plot(round_list,svm_results_scaled, color ='g')
plt.show()
