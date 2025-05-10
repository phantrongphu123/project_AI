Flow Free Solver & Benchmarker - Galaxy Edition
Ứng dụng giải đố Flow Free  bằng Python với giao diện Tkinter, hỗ trợ nhiều thuật toán giải, tính năng benchmark và hiển thị biểu đồ so sánh hiệu suất.
## Giới thiệu
Flow Free là một trò chơi giải đố logic, mục tiêu là kết nối các cặp điểm màu giống nhau trên một lưới ô vuông sao cho các đường đi không cắt nhau và lấp đầy toàn bộ lưới. Dự án này cung cấp một công cụ để:
1. Giải các câu đố Flow Free sử dụng nhiều thuật toán khác nhau.
2. Hiển thị trực quan quá trình giải và kết quả trên giao diện người dùng.
3. Đánh giá và so sánh hiệu suất của các thuật toán thông qua một bộ benchmark.
4. Tối ưu hóa heuristic cho thuật toán A*.
5. Hiển thị biểu đồ kết quả benchmark (nếu có Matplotlib).
## Tính năng nổi bật
* Giao diện người dùng đồ họa (GUI): Được xây dựng bằng Tkinter với chủ đề "Galaxy" tùy chỉnh.
* Nhiều thuật toán giải:
    * Backtracking (Quay lui tối ưu)
    * Breadth-First Search (BFS - Tìm kiếm theo chiều rộng)
    * A* Search (Tìm kiếm A sao) với các heuristic tùy chọn.
    * Constraint Programming (CP - Lập trình ràng buộc) sử dụng Google OR-Tools (nếu được cài đặt).
* Lựa chọn Puzzle linh hoạt: Tải các puzzle mẫu được định nghĩa sẵn theo độ khó.
* Hiển thị trực quan: Vẽ đường đi của các màu trên lưới, với hiệu ứng hoạt ảnh khi giải xong.
* Đánh giá Heuristic (cho A*): Chạy thử nghiệm trên puzzle hiện tại với các heuristic khác nhau để tìm ra heuristic hiệu quả nhất.
* Benchmark Suite:
    * Chạy tự động nhiều puzzle với các thuật toán đã chọn.
    * Thiết lập giới hạn thời gian và số trạng thái cho mỗi lần chạy.
    * Hiển thị kết quả benchmark trong bảng.
    * Lưu kết quả benchmark ra file CSV.
* Biểu đồ Benchmark (Yêu cầu Matplotlib):
    * Hiển thị biểu đồ so sánh thời gian giải của các thuật toán cho puzzle hiện tại.
    * Hiển thị biểu đồ số lượng thuật toán giải được cho mỗi puzzle trong một độ khó.
* Chạy từ dòng lệnh: Hỗ trợ chạy benchmark suite hoàn toàn từ dòng lệnh với các tùy chọn cấu hình.
## Yêu cầu hệ thống
* Python 3.7+
* Tkinter (thường đi kèm với Python)
* Tùy chọn (để có đầy đủ tính năng):
    * matplotlib: Cho chức năng vẽ biểu đồ.
    * ortools: Cho thuật toán Constraint Programming (CP).
```
## Hướng dẫn sử dụng (GUI)
1. Chọn độ khó và puzzle từ menu.
2. Chọn thuật toán và heuristic (nếu dùng A*).
3. Nhấn “Giải & Vẽ” để giải puzzle.
4. Nhấn “Reset Puzzle” để làm mới.
5. Chạy benchmark bằng nút tương ứng.
6. Xem biểu đồ nếu đã chạy benchmark và có matplotlib.
## Các thuật toán được triển khai
* Backtracking (Quay lui): DFS tối ưu, ưu tiên theo khoảng cách Manhattan.
* BFS: Tìm kiếm theo chiều rộng.
* A* Search: Dùng heuristic để dẫn đường tìm kiếm.
* Constraint Programming: Dùng Google OR-Tools, mô hình hóa bằng ràng buộc.
## Heuristics cho A*
* Manhattan Sum
* Manhattan Max
* Manhattan Avg + Incomplete Penalty
## Định dạng Puzzle đầu vào
Puzzle là chuỗi ký tự:
* . đại diện ô trống
* 1-9 và A-Z là các điểm đầu/cuối của các màu
* Mỗi màu phải có đúng hai điểm
Ví dụ:
```
1.2
...
1.2
```
Hoặc:
```
1.2.5
..3.4
.....
.2.5.
.134.
```
## Cấu trúc mã nguồn (Tổng quan)
* Import thư viện
* Hàm tiện ích: get_neighbors, parse_puzzle_extended, v.v.
* Heuristic: h_manhattan_sum, h_manhattan_max, ...
* Thuật toán: solve_backtracking, solve_cp, solve_bfs, solve_astar
* Dữ liệu puzzle mẫu: PUZZLES
* GUI: FlowFreeApp với các phương thức khởi tạo, sự kiện, hiển thị, giải puzzle
## Link github:https://github.com/phantrongphu123/project_AI
