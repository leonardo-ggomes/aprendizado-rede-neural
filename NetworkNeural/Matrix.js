class Matrix{
    
    constructor(rows, cols){
        this.rows = rows
        this.cols = cols
        
        this.data = Array(rows).fill().map(() => Array(cols).fill(0));
        
    }

    randomize(){
        this.data = this.data.map(row => 
            row.map(() => Math.random() * 2 - 1)
        );
    }    

    map(func){        
      
        this.data = this.data.map((row, r_index) => {           
            return row.map((c) => {
                return func(c)
           })          
        })

        return this
    }

    static map(A, func){        
        let m = new Matrix(A.rows, A.cols);

        m.data = A.data.map((row) => {           
            return row.map((c) => {
                return func(c)
           })          
        })

        return m
    }

    static arrayToMatrix(arr){
        let m = new Matrix(arr.length, 1);
        m.data = m.data.map((row, i) => [arr[i]]);
        return m;
    }

    static subtract(A, B){
        const m = new Matrix(A.rows, A.cols)

        m.data = m.data.map((row, i) =>
            row.map((_, j) => A.data[i][j] - B.data[i][j])
        )

        return m
    }

    static multiply(A, B) {
        
        let m = new Matrix(A.rows, B.cols);

        m.data = m.data.map((row, i) => {

            return row.map((colValue, j) => {
                let sum = 0;
                for (let k = 0; k < A.cols; k++) {
                    sum += A.data[i][k] * B.data[k][j];
                }
                return sum;
            });

        });


        return m;
    }

    static outerProduct(A, B) {

        const m = new Matrix(A.rows, B.cols)

        for (let i = 0; i < A.rows; i++) {
            for (let j = 0; j < B.cols; j++) {
                m.data[i][j] = A.data[i][0] * B.data[0][j]
            }
        }

        return m
    }

    static add(A, B){
        const m = new Matrix(A.rows, A.cols)

        m.data = m.data.map((row, i) =>
            row.map((_, j) => A.data[i][j] + B.data[i][j])
        )

        return m
    }

    static hadamard(A, B){
        const m = new Matrix(A.rows, A.cols)

        A.data.forEach((row, r) => {
            row.forEach((_, c) => {
                m.data[r][c] = A.data[r][c] * B.data[r][c]
            })
        })

        return m
    }

    static multiply_scalar(A, factor){
        const m = new Matrix(A.rows, A.cols)

        A.data.forEach((row, r) => {
            row.forEach((_, c) => {
                m.data[r][c] = A.data[r][c] * factor
            })
        })

        return m
    }

    static transpose(A){
        const m = new Matrix(A.cols, A.rows)

        A.data.map((row, r) => {
            row.map((col, c) => {
                m.data[c][r] = col
            })
        })

        return m
    } 

   
    print(){
        console.table(this.data)
    }

}

export default Matrix