import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Thiết lập bài toán mẫu
n = 20  # số biến
m = 100  # số điều kiện
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Giải bài toán bằng CVXPY để tìm giá trị tối ưu
x_min = cp.Variable(n)
objective = cp.Minimize(cp.max(A @ x_min + b))
problem = cp.Problem(objective)
problem.solve()

f_min = problem.value
print(f"Optimal value is {f_min:.4f}.\n")

# Điểm khởi tạo
x_1 = np.zeros(n)

# Hàm subgradient method với bước cố định
def sgm_pwl_const_step_length(A, b, x1, R, gamma, MAX_ITERS):
    f = [np.inf]
    fbest = [np.inf]
    x = x1
    for k in range(MAX_ITERS):
        # Tính gradient phụ
        fval = np.max(A @ x + b)
        ind = np.argmax(A @ x + b)
        g = A[ind, :]
        
        # Kích thước bước cố định
        alpha = gamma / np.linalg.norm(g)
        
        # Cập nhật hàm mục tiêu và giá trị tốt nhất
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))
        
        # Cập nhật biến
        x = x - alpha * g
    
    hist = [f, fbest]
    return x, hist

# Thiết lập tham số và thực hiện phương pháp bội gradient phụ với các bước cố định
MAX_ITERS = 3000
gammas = [0.05, 0.01, 0.005]
results = []

for gamma in gammas:
    x, hist = sgm_pwl_const_step_length(A, b, x_1, 10, gamma, MAX_ITERS)
    results.append(hist)

# Thiết lập dữ liệu đồ thị
iters = np.arange(1, MAX_ITERS + 1)
fbest1 = np.array(results[0][1])[:MAX_ITERS]
fbest2 = np.array(results[1][1])[:MAX_ITERS]
fbest3 = np.array(results[2][1])[:MAX_ITERS]

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.semilogy(iters, fbest1 - f_min, 'r-.', linewidth=1.5, label=r'$\gamma = 0.05$')
plt.semilogy(iters, fbest2 - f_min, 'g--', linewidth=1.5, label=r'$\gamma = 0.01$')
plt.semilogy(iters, fbest3 - f_min, 'b-', linewidth=1.5, label=r'$\gamma = 0.005$')

plt.xlabel('k')
plt.ylabel(r'$f_{\mathrm{best}}^{(k)} - f^*$')
plt.legend()
plt.title('Hội tụ của phương pháp dưới đạo hàm với độ dài cỡ bước không đổi')
plt.grid(True)
plt.show()
