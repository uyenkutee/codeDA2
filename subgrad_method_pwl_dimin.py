
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

# Hàm subgradient method với bước giảm dần
def sgm_pwl_dimin_step(A, b, x1, a, step_rule, MAX_ITERS):
    f = [np.inf]
    fbest = [np.inf]
    x = x1
    for k in range(1, MAX_ITERS + 1):
        # Tính gradient phụ
        fval = np.max(A @ x + b)
        ind = np.argmax(A @ x + b)
        g = A[ind, :]
        
        # Chọn kích thước bước dựa trên quy tắc
        if step_rule == 'sqrt':
            alpha = a / np.sqrt(k)
        elif step_rule == 'linear':
            alpha = a / k
        
        # Cập nhật hàm mục tiêu và giá trị tốt nhất
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))
        
        # Cập nhật biến
        x = x - alpha * g
    
    hist = [f, fbest]
    return x, hist

# Thiết lập tham số và thực hiện phương pháp bội gradient phụ với các bước giảm dần
MAX_ITERS = 3000
x1, hist1 = sgm_pwl_dimin_step(A, b, x_1, 0.1, 'sqrt', MAX_ITERS)   # α_k = 0.1 / √k
x2, hist2 = sgm_pwl_dimin_step(A, b, x_1, 1, 'sqrt', MAX_ITERS)     # α_k = 1 / √k
x3, hist3 = sgm_pwl_dimin_step(A, b, x_1, 1, 'linear', MAX_ITERS)   # α_k = 1 / k
x4, hist4 = sgm_pwl_dimin_step(A, b, x_1, 10, 'linear', MAX_ITERS)  # α_k = 10 / k

# Thiết lập dữ liệu đồ thị
iters = np.arange(1, MAX_ITERS + 1)
fbest1 = np.array(hist1[1])
fbest2 = np.array(hist2[1])
fbest3 = np.array(hist3[1])
fbest4 = np.array(hist4[1])

# Vẽ đồ thị 1: Quy tắc bước căn bậc hai
plt.figure(figsize=(10, 6))
plt.semilogy(iters, fbest1[:MAX_ITERS] - f_min, 'b-', linewidth=1.5, label=r'$\alpha_k = 0.1 / \sqrt{k}$')  # Nét liền
plt.semilogy(iters, fbest2[:MAX_ITERS] - f_min, 'g--', linewidth=1.5, label=r'$\alpha_k = 1 / \sqrt{k}$')  # Nét đứt

plt.xlabel('k')
plt.ylabel(r'$f_{\mathrm{best}}^{(k)} - f^*$')
plt.legend()
plt.title('Dưới đạo hàm với kích thước bước giảm dần')
plt.grid(True)
plt.show()

# Vẽ đồ thị 2: Quy tắc bước tuyến tính
plt.figure(figsize=(10, 6))
plt.semilogy(iters, fbest3[:MAX_ITERS] - f_min, 'r-', linewidth=1.5, label=r'$\alpha_k = 1 / k$')  # Nét liền
plt.semilogy(iters, fbest4[:MAX_ITERS] - f_min, 'k--', linewidth=1.5, label=r'$\alpha_k = 10 / k$')  # Nét đứt

plt.xlabel('k')
plt.ylabel(r'$f_{\mathrm{best}}^{(k)} - f^*$')
plt.legend()
plt.title('Dưới đạo hàm với quy tắc bước tổng bình phương hữu hạn nhưng không khả tổng')
plt.grid(True)
plt.show()

