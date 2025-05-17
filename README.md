![si-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/fd7f2f6c-8969-4376-9b74-da75fd9be589)# Báo cáo Đồ án: Xây dựng Game Flow Free và Ứng dụng các Thuật toán Tìm kiếm AI

## 1. Giới thiệu

Bài toán Flow Free là một trò chơi giải đố logic phổ biến, thử thách người chơi nối các cặp điểm màu trên một lưới ô vuông. Mục tiêu là tạo ra các đường đi (flows) sao cho mỗi đường nối đúng cặp màu của nó, các đường đi của màu khác nhau không được giao nhau, và toàn bộ lưới phải được lấp đầy. Đồ án này được thực hiện với mục tiêu chính là xây dựng một phiên giải đố của game Flow Free có giao diện đồ họa, đồng thời triển khai và ứng dụng các thuật toán tìm kiếm trong trí tuệ nhân tạo để tự động hóa việc tìm lời giải. Bên cạnh đó, một phần quan trọng của đồ án là cung cấp công cụ để so sánh và đánh giá hiệu năng của các thuật toán này trên nhiều màn chơi với độ khó đa dạng, cũng như minh họa trực quan quá trình giải của chúng.

## 2. Các tính năng chính

Dự án nổi bật với giao diện người dùng đồ họa được xây dựng bằng Tkinter, mang lại trải nghiệm trực quan và dễ sử dụng. Người dùng có thể dễ dàng lựa chọn các màn chơi từ nhiều mức độ khó khác nhau, từ "Tiny" đến "Very Hard". Điểm cốt lõi của dự án là việc triển khai một loạt các thuật toán AI bao gồm Backtracking, Breadth-First Search (BFS), A* Search với nhiều tùy chọn heuristic, Constraint Programming (CP) sử dụng  OR-Tools, Q-Learning, Simulated Annealing, và AND-OR Search. Các thuật toán này cho phép tự động tìm lời giải cho puzzle đã chọn, và kết quả đường đi sẽ được vẽ trực quan trên lưới, thậm chí có hiệu ứng hoạt ảnh sinh động. Một bộ công cụ Benchmarking Suite mạnh mẽ cũng được tích hợp, cho phép chạy tự động các thuật toán trên một tập hợp puzzle, ghi nhận chi tiết thời gian giải, số trạng thái duyệt, và kết quả cuối cùng. Dữ liệu benchmark này có thể được lưu dưới dạng file CSV và hiển thị thành các biểu đồ so sánh hiệu năng nếu thư viện Matplotlib được cài đặt. Ngoài ra, dự án còn cung cấp các công cụ chuyên biệt để đánh giá hiệu quả của các hàm heuristic khác nhau cho thuật toán A* và các cấu hình siêu tham số cho Q-Learning.

## 3. Công nghệ sử dụng

Dự án được phát triển chủ yếu bằng ngôn ngữ Python 3.x. Giao diện người dùng đồ họa được xây dựng bằng thư viện Tkinter. Đối với phần trí tuệ nhân tạo và logic giải quyết vấn đề, Google OR-Tools được sử dụng cho thuật toán Constraint Programming (đây là một thư viện tùy chọn). Nhiều thư viện tiện ích chuẩn của Python như `copy`, `time`, `threading`, `heapq`, `collections.deque` cũng đóng vai trò quan trọng. Thư viện Matplotlib (tùy chọn) được dùng để trực quan hóa dữ liệu benchmark thông qua các biểu đồ.

## 4. Cài đặt và Chạy chương trình

Để chạy dự án, yêu cầu hệ thống cần có Python 3.7 trở lên và trình quản lý gói pip. 
Hầu hết các thư viện chuẩn đã có sẵn. Tuy nhiên, để trải nghiệm đầy đủ các tính năng, đặc biệt là CP Solver và vẽ biểu đồ, người dùng cần cài đặt Google OR-Tools và Matplotlib thông qua pip: `pip install ortools matplotlib`. Chương trình vẫn có thể hoạt động mà không có chúng, nhưng các chức năng tương ứng sẽ bị vô hiệu hóa.

Để khởi chạy ứng dụng GUI, chỉ cần thực thi file chính: `python group06_flowfree.py`.

Ngoài ra, dự án hỗ trợ chế độ benchmark chạy từ dòng lệnh, rất hữu ích cho việc kiểm thử tự động. Ví dụ: `python group06_flowfree.py --run_benchmark --algorithms "A*,BFS" --puzzles "Easy (5x5)" --output_csv "results.csv"`. Các tham số như thuật toán, độ khó, giới hạn thời gian, giới hạn trạng thái và tên file output đều có thể tùy chỉnh.

## 5. Mô hình hóa bài toán Flow Free

Việc áp dụng hiệu quả các thuật toán tìm kiếm đòi hỏi một mô hình hóa bài toán rõ ràng. Trong dự án này, một **trạng thái** được biểu diễn bởi một lưới 2D lưu trữ thông tin ô trống, điểm đầu/cuối, hoặc một phần đường đi đã vẽ, cùng với một cấu trúc dữ liệu theo dõi chi tiết các đường đi hiện tại của mỗi màu và trạng thái hoàn thành của chúng. **Hành động** là việc mở rộng một đường đi chưa hoàn chỉnh sang một ô kề cận hợp lệ, tức là ô đó nằm trong lưới, trống hoặc là điểm cuối của màu đang vẽ, và không gây tự cắt. **Hàm chuyển tiếp** sẽ cập nhật trạng thái lưới và đường đi sau mỗi hành động. **Trạng thái ban đầu** là lưới chỉ có các điểm màu. Một trạng thái được coi là **mục tiêu** khi tất cả các màu đã được nối đúng cách và toàn bộ lưới được lấp đầy. Chi phí hành động thường là 1 cho mỗi ô được vẽ, mặc dù trong Flow Free, tất cả lời giải hợp lệ thường có cùng tổng chi phí (số ô được lấp đầy).

## 6. Các thuật toán tìm kiếm được triển khai và ứng dụng

Dự án đã triển khai và tích hợp một loạt các thuật toán AI để giải quyết bài toán Flow Free, mỗi thuật toán có cách tiếp cận và đặc điểm riêng:

**6.1. Backtracking:** Thuật toán này hoạt động dựa trên nguyên lý thử-sai có hệ thống. Nó xây dựng giải pháp từng bước cho mỗi màu, ưu tiên mở rộng một màu cho đến khi hoàn thành hoặc gặp ngõ cụt. Nếu không thể đi tiếp hoặc các bước sau không dẫn đến lời giải, nó sẽ "quay lui" lại lựa chọn trước đó và thử một hướng đi khác. Mặc dù đơn giản và không tốn nhiều bộ nhớ, hiệu suất của Backtracking phụ thuộc nhiều vào thứ tự duyệt và có thể chậm với các puzzle phức tạp.
![Backtracking](https://github.com/user-attachments/assets/39d883e6-3ff7-407f-a53f-2e23468a2fb0)

**6.2. Breadth-First Search (BFS):** BFS khám phá không gian trạng thái theo từng lớp, đảm bảo tìm ra lời giải nông nhất nếu tồn tại. Nó sử dụng một hàng đợi để quản lý các trạng thái (bao gồm lưới, thông tin các đường đi, và màu đang được mở rộng) và một tập `visited` để tránh chu trình. BFS hoàn chỉnh nhưng đòi hỏi không gian bộ nhớ lớn.
![BFS](https://github.com/user-attachments/assets/8e6fff8b-8085-4b9e-89ba-ec2ffdd41d16)
***6.3. A* Search:** Đây là một thuật toán tìm kiếm có thông tin, sử dụng hàm đánh giá `f(n) = g(n) + h(n)` để hướng dẫn quá trình tìm kiếm. `g(n)` là chi phí thực tế (số ô đã vẽ) và `h(n)` là chi phí ước lượng đến đích. Dự án triển khai nhiều hàm heuristic như tổng khoảng cách Manhattan, khoảng cách Manhattan lớn nhất, và một biến thể kết hợp trung bình Manhattan với hình phạt cho các đường chưa hoàn thành. A* hiệu quả hơn BFS nếu có heuristic tốt và đảm bảo tối ưu nếu heuristic là "admissible".
![a_start](https://github.com/user-attachments/assets/1dde7240-65b6-4a4c-a59f-3f03cfb75f88)

**6.4. Constraint Programming (CP) với OR-Tools:** Cách tiếp cận này mô hình hóa Flow Free như một bài toán thỏa mãn ràng buộc (CSP). Các biến Boolean `is_path[r][c][k]` biểu thị ô `(r,c)` có thuộc đường đi của màu `k` hay không. Các ràng buộc bao gồm: mỗi ô chỉ thuộc một đường đi, các điểm đầu/cuối phải được gán đúng màu, và tính liên tục của đường đi (mỗi ô trên đường đi phải có đúng 2 hàng xóm cùng màu, trừ điểm đầu/cuối chỉ có 1). OR-Tools sau đó được sử dụng để tìm một phép gán thỏa mãn tất cả ràng buộc. Đây là một phương pháp mạnh mẽ, khai báo, nhưng đòi hỏi việc tái tạo đường đi từ kết quả lưới.
![CP](https://github.com/user-attachments/assets/3a942542-745b-4342-8c43-02211c9246e0)

**6.5. Q-Learning:** Thuật toán học tăng cường này cho phép một tác nhân học chiến lược giải thông qua tương tác thử-sai với môi trường. Trạng thái được định nghĩa bởi lưới và màu đang hoạt động, hành động là việc chọn ô tiếp theo. Hàm Q `Q(s,a)` ước lượng chất lượng của hành động. Phần thưởng được thiết kế để khuyến khích hoàn thành các màu và lấp đầy lưới. Sau nhiều episodes học, Q-table được sử dụng để đưa ra quyết định tối ưu.
![q-learning](https://github.com/user-attachments/assets/4e394393-a1e5-405a-b7c8-56744dc84ac3)


**6.6. Simulated Annealing:** Thuật toán tối ưu hóa này bắt đầu với một giải pháp ngẫu nhiên và cố gắng cải thiện dần bằng cách thực hiện các thay đổi nhỏ. Nó có khả năng chấp nhận các bước đi "tệ hơn" với một xác suất nhất định (giảm dần theo "nhiệt độ") để tránh bị kẹt ở các điểm tối ưu cục bộ. Hàm năng lượng đánh giá chất lượng giải pháp dựa trên số ô trống, khoảng cách chưa hoàn thành, v.v.
![simulated](https://github.com/user-attachments/assets/5165609e-b457-499a-8e78-b4bbd0aef0e6)


**6.7. AND-OR Search:** Thuật toán này được thiết kế cho các bài toán có thể phân rã. Trong Flow Free, mỗi trạng thái lưới có thể xem là một nút OR, nơi tác nhân chọn một màu và một hướng đi. Vì game là tất định, hành động đó dẫn đến một trạng thái mới duy nhất. Quá trình tìm kiếm tương tự DFS có cấu trúc, nhằm tìm một chuỗi hành động dẫn đến mục tiêu.
![and_or-search](https://github.com/user-attachments/assets/8c245352-b0c5-4455-8755-be0fb3493059)


## 7. Tổng quan Giao diện Người dùng (GUI)

Giao diện người dùng của ứng dụng được thiết kế trực quan với các thành phần chính: khu vực chọn puzzle theo độ khó và chỉ số; khu vực cấu hình thuật toán cho phép chọn thuật toán, heuristic (cho A*), hoặc cấu hình (cho Q-Learning); khu vực hiển thị lưới game trung tâm; khu vực hành động với các nút "Giải & Vẽ", "Reset Puzzle", "Chạy Benchmark Suite", và "Hiển thị Biểu đồ"; cuối cùng là thanh trạng thái cung cấp thông tin cập nhật.

## 8. Benchmarking và Đánh giá

Một trong những điểm mạnh của dự án là bộ công cụ benchmarking. Người dùng có thể cấu hình để chạy nhiều thuật toán trên các tập puzzle khác nhau, với các giới hạn về thời gian và số trạng thái. Kết quả chi tiết về thời gian, trạng thái duyệt, và tình trạng giải được ghi nhận và có thể lưu ra file CSV. Nếu Matplotlib được cài đặt, kết quả này có thể được trực quan hóa qua các biểu đồ so sánh hiệu năng, ví dụ như thời gian giải của các thuật toán trên từng puzzle, số thuật toán giải được mỗi puzzle, hay tổng số puzzle mỗi thuật toán giải được. Điều này cung cấp cái nhìn sâu sắc về điểm mạnh và yếu của từng phương pháp.

## 9. Hướng phát triển tương lai

Dự án vẫn còn nhiều tiềm năng phát triển. Về mặt GUI, có thể bổ sung tính năng cho người dùng tự vẽ, lưu/tải tiến trình, hoặc tạo puzzle ngẫu nhiên. Về thuật toán, có thể triển khai các biến thể nâng cao hơn như IDA*, tối ưu hóa các thuật toán hiện tại, hoặc nghiên cứu sâu hơn về thiết kế heuristic. Việc áp dụng học tăng cường sâu (Deep RL) cũng là một hướng thú vị. Ngoài ra, phân tích lý thuyết về độ phức tạp của Flow Free và hỗ trợ các biến thể game như Flow Free Bridges hay Hexes cũng là những cải tiến đáng giá.

## 10. Tác giả

*   (Điền tên các thành viên trong nhóm của bạn vào đây)
