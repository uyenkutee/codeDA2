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

# Hàm subgradient method với quy tắc bước Nonsummable diminishing step lengths
def sgm_nonsumm_dimin_step_length(A, b, x1, gamma_0, step_rule, MAX_ITERS):
    f = [np.inf]
    fbest = [np.inf]
    x = x1
    for k in range(1, MAX_ITERS + 1):
        # Tính gradient phụ
        fval = np.max(A @ x + b)
        ind = np.argmax(A @ x + b)
        g = A[ind, :]

        # Chọn kích thước bước dựa trên quy tắc nonsummable diminishing step lengths
        if step_rule == 'sqrt':
            gamma_k = gamma_0 / np.sqrt(k)
        elif step_rule == 'linear':
            gamma_k = gamma_0 / k
        
        alpha = gamma_k / np.linalg.norm(g)

        # Cập nhật hàm mục tiêu và giá trị tốt nhất
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # Cập nhật biến
        x = x - alpha * g

    hist = [f, fbest]
    return x, hist

# Thiết lập tham số và thực hiện phương pháp bội gradient phụ với quy tắc nonsummable diminishing step lengths
MAX_ITERS = 3000
x5, hist5 = sgm_nonsumm_dimin_step_length(A, b, x_1, 0.1, 'sqrt', MAX_ITERS)  # \gamma_k = 0.1 / \sqrt{k}
x7, hist7 = sgm_nonsumm_dimin_step_length(A, b, x_1, 1, 'linear', MAX_ITERS)  # \gamma_k = 1 / k

# Thiết lập dữ liệu đồ thị
iters = np.arange(1, MAX_ITERS + 1)
fbest5 = np.array(hist5[1])
fbest7 = np.array(hist7[1])

# Vẽ đồ thị: Quy tắc bước Nonsummable diminishing step lengths
plt.figure(figsize=(10, 6))
plt.semilogy(iters, fbest5[:MAX_ITERS] - f_min, 'b', linewidth=1.5, label=r'$\gamma_k = 0.1 / \sqrt{k}$')
plt.semilogy(iters, fbest7[:MAX_ITERS] - f_min, 'r-.', linewidth=1.5, label=r'$\gamma_k = 1 / k$')

plt.xlabel('k')
plt.ylabel(r'$f_{\mathrm{best}}^{(k)} - f^*$')
plt.legend()
plt.title('Dưới đạo hàm với độ dài bước giảm dần nhưng không khả tổng')
plt.grid(True)
plt.show()