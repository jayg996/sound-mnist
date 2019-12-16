from matplotlib import pyplot as plt

# bar graph
performance = [0.585, 0.865, 0.869, 0.835, 0.46, 0.81]
method = ['LDA', 'QDA', 'NN', 'SVM', 'DT', 'RF']
x = [0, 1, 2, 3, 4, 5]

plt.bar(method, performance, width=0.5, color='c')
for i in range(len(method)):
    plt.text(x=x[i] - 0.2, y=performance[i] + 0.02, s=performance[i], size=10)

plt.xlabel('Method')
plt.ylabel('Classification Accuracy')
plt.title('Performance of model with best hyperparameter')
plt.ylim([0.0, 1.0])
plt.savefig('performance_bar.png')
plt.show()
plt.close()

# comparison bar graph
before_scale = [0.274, 0.429, 0.647, 0.305, 0.46, 0.804]

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]

value_a_x = create_x(2, 0.8, 1, 6)
value_b_x = create_x(2, 0.8, 2, 6)
ax = plt.subplot()
ax.bar(value_a_x, before_scale, color='C1')
ax.bar(value_b_x, performance, color='c')
middle_x = [(a+b)/2 for (a, b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(method)
for i in range(len(method)):
    plt.text(x=value_a_x[i] - 0.4, y=before_scale[i] + 0.01, s=before_scale[i], size=8)
    plt.text(x=value_a_x[i] + 0.45, y=performance[i] + 0.01, s=performance[i], size=8)
plt.legend(('Befor scaling', 'Scaling'))
plt.title('Performance comparison by scaling')
plt.xlabel('Method')
plt.ylabel('Classification Accuracy')
plt.ylim([0.0, 1.0])
plt.savefig('comparison_bar.png')
plt.show()
plt.close()

# real test

real_test = [0.6, 0.9, 0.9, 0.8, 0.4, 0.9]

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]

value_a_x = create_x(2, 0.8, 1, 6)
value_b_x = create_x(2, 0.8, 2, 6)
ax = plt.subplot()
ax.bar(value_a_x, real_test, color='C4')
ax.bar(value_b_x, performance, color='c')
middle_x = [(a+b)/2 for (a, b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(method)
for i in range(len(method)):
    plt.text(x=value_a_x[i] - 0.4, y=real_test[i] + 0.01, s=real_test[i], size=8)
    plt.text(x=value_a_x[i] + 0.45, y=performance[i] + 0.01, s=performance[i], size=8)
plt.legend(('Real test', 'Test'))
plt.title('Performance of real test')
plt.xlabel('Method')
plt.ylabel('Classification Accuracy')
plt.ylim([0.0, 1.0])
plt.savefig('real_test_bar.png')
plt.show()
plt.close()
