import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class Procesador {
    // Juego de instrucciones. código de operación
    static int add = 1;
    static int sub = 2;
    static int ld = 3;
    static int sd = 4;
    static int mult = 5;

    // Capacidad de las estructuras de almacenamiento es ilimitada pero se pone un máximo
    static int REG = 16; /* Número de registros */
    static int DAT = 32; /* Tamaño de memoria de datos */
    static int INS = 32; /* Tamaño de memoria de instrucciones */

    /* Códigos para las UF */
    static int TOTAL_UF = 3; /* total de UF simuladas */
    static int ALU = 0; /* código para una UF de suma/resta */
    static int MEM = 1; /* código para UF de carga/almacenamiento */
    static int MULT = 2; /* código para UF de multiplicación */

    /* Ciclos de ejecución de UF */
    static int CICLOS_MEM = 2; /* Carga y almacenamiento */
    static int CICLOS_ALU = 3; /* Suma y Resta */
    static int CICLOS_MULT = 5; /* Multiplicación */

    /* Etapas de procesamiento de las instrucciones en ROB */
    static int ISS = 1; /* Instrucción emitida */
    static int EX = 2; /* Instrucción en ejecucion EX */
    static int WB = 3; /* Fase WB realizada */

    /* Estructuras de datos */ //Instrucción

    Registro[] BancoRegistros = new Registro[REG]; // cada linea de los registros
    Instruccion[] ListaInstrucciones = new Instruccion[INS];

    int[] MemoriaDatos = new int[DAT];

    public class Instruccion {
        int cod; /* operación a realizar */
        int rd; /* registro destino */
        int rs; /* registro fuente op1 */
        int rt; /* registro fuente op2 */
        int inmediato; /* Dato inmediato */

        public Instruccion(int cod, int rd, int rs, int rt, int inmediato) { //Ld f2,0(x1) ld rt, inm(rs)
            this.cod = cod;
            this.rd = rd;
            this.rs = rs;
            this.rt = rt;
            this.inmediato = inmediato;
        }

        public void pr() {
            System.out.println("codigo: " + this.cod + ", rd: " + this.rd + ", rs: " + this.rs +
                    ", rt: " + this.rt + ", inmediato: " + this.inmediato);
        }
    }

    class Registro { // reg_t Registro del banco de registros

        int contenido; /* contenido */
        int ok; /* contenido valido (1) o no (0) */
        int clk_tick_ok; /* ciclo de reloj cuando se valida ok. */
        int TAG_ER; /* Si ok = 0, etiqueta línea de ER donde está almacenada la instrucción que generara ese resultado. */

        /* último rd */
        public Registro(int contenido, int ok, int clk_tick_ok, int TAG_ER) {
            this.contenido = contenido;
            this.ok = ok;
            this.clk_tick_ok = clk_tick_ok;
            this.TAG_ER = TAG_ER;
        }

        public String pr() {
            System.out.println();
            return "contenido: " + this.contenido + ", ok: " + this.ok + ", clk_tick_ok: " + this.clk_tick_ok +
                    ", TAG_ER: " + this.TAG_ER;
        }


    }

    class EstacionReserva { //ER_T EstacionReserva
        int busy; /* contenido de la línea válido (1) o no (0) */
        int operacion; /* operación a realizar en UF (suma,resta,mult,lw,sw) */
        int opa; /* valor Vj*/
        int opa_ok; /* Qj, (1) opa válido o no (0)*/
        int clk_tick_ok_a; /* ciclo de reloj donde opa_ok = 1*/
        int opb; /* Valor Vk */
        int opb_ok; /* Qk, (1) válido o (0) no válido */
        int clk_tick_ok_b; /* ciclo de reloj donde opb_ok = 1*/
        int inmediato; /*utilizado para las instrucciones lw/sw */
        int TAG_ER; /* etiqueta identificativa de estación de reserva */

        public EstacionReserva(int busy, int operacion, int opa, int opa_ok, int clk_tick_ok_a, int opb, int opb_ok, int clk_tick_ok_b, int inmediato, int TAG_ER) {
            this.busy = busy;
            this.operacion = operacion;
            this.opa = opa;
            this.opa_ok = opa_ok;
            this.clk_tick_ok_a = clk_tick_ok_a;
            this.opb = opb;
            this.opb_ok = opb_ok;
            this.clk_tick_ok_b = clk_tick_ok_b;
            this.inmediato = inmediato;
            this.TAG_ER = TAG_ER;
        }

        public void pr() {
            System.out.println("busy: " + this.busy + ", operacion: " + this.operacion + ", opa: " + this.opa +
                    ", opa_ok: " + this.opa_ok + ", clk_tick_ok_a: " + this.clk_tick_ok_a + ", opb: " + this.opb +
                    ", opb_ok: " + this.opb_ok + ", clk_tick_ok_b: " + this.clk_tick_ok_b + ", inmediato: " +
                    this.inmediato + ", TAG_ER: " + this.TAG_ER);
        }
    }

    class UnidadFuncional {
        int uso; /* Indica si UF está utilizada (1) o no (0) */
        int cont_ciclos; /* indica ciclos consumidos por la UF */
        int TAG_ER; /* Línea de ER donde está la instrucción que ha generado esta operación */
        int opa; /* valor opa (en sd contiene dato a escribir en memoria */
        int opb; /* valor opb (en ld y sd contiene registro que contiene parte dirección de memoria */
        int operacion; /* se utiliza para indicar operacion a realizar add/sub y lw/sw o mult */
        int res; /* resultado */
        int res_ok; /* resultado valido (1) */
        int clk_tick_ok; /* ciclo de reloj cuando se valida res_ok */

        public UnidadFuncional(int uso, int cont_ciclos, int TAG_ER, int opa, int opb, int operacion, int res, int res_ok, int clk_tick_ok) {
            this.uso = uso;
            this.cont_ciclos = cont_ciclos;
            this.TAG_ER = TAG_ER;
            this.opa = opa;
            this.opb = opb;
            this.operacion = operacion;
            this.res = res;
            this.res_ok = res_ok;
            this.clk_tick_ok = clk_tick_ok;
        }

        public void pr() {
            System.out.println("Uso: " + this.uso + ", cont_ciclos: " + this.cont_ciclos + ", TAG_ER: " + this.TAG_ER + ", opa: " + this.opa + ", opb: " + this.opb +
                    ", operacion: " + this.operacion + ", res: " + this.res + ", res_ok: " + this.res_ok + ", clk_tick_ok: " + this.clk_tick_ok);
        }

    } //TABLA QUE CONTIENE LOS CICLOS QUE TARDAN LAS INSTRUCCIONES EN EJECUTAR EL LA ETAPA EX Y SU INFORMACION CORRESPONDIENTE

    public static void main(String[] args) {
        Procesador procesador = new Procesador();
    }
    public int leer_programa(Instruccion[] memoriaInstrucciones) {
        int i = 0;
        try {
            BufferedReader br = new BufferedReader(new FileReader("instrucciones7.txt"));
            List<Instruccion> instrucciones = new ArrayList<>();
            String linea;
            while ((linea = br.readLine()) != null) {
                String[] partes = linea.split(",");
                if (partes[0].equals("fadd")) {
                    int f1 = Integer.parseInt(partes[1].substring(1)); // Quita la "f" del primer registro
                    int f2 = Integer.parseInt(partes[2].substring(1));
                    int f3 = Integer.parseInt(partes[3].substring(1));
                    int inmediato = 0;
                    Instruccion ins = new Instruccion(1, f1, f2, f3, inmediato);
                    memoriaInstrucciones[i]=ins;
                    i++;
                } else if (partes[0].equals("fsub")) {
                    int f1 = Integer.parseInt(partes[1].substring(1)); // Quita la "f" del primer registro
                    int f2 = Integer.parseInt(partes[2].substring(1));
                    int f3 = Integer.parseInt(partes[3].substring(1));
                    int inmediato = 0;
                    Instruccion ins = new Instruccion(2, f1, f2, f3, inmediato);
                    memoriaInstrucciones[i]=ins;
                    i++;
                }else if (partes[0].equals("ld")) {
                    int f2 = Integer.parseInt(partes[1].substring(1)); // Quita la "f" del segundo registro
                    int x3 = Integer.parseInt(partes[3].substring(1));
                    int inmediato = Integer.parseInt(partes[2]);
                    Instruccion ins = new Instruccion(3, f2, -1, x3, inmediato);
                    memoriaInstrucciones[i]=ins;
                    i++;
                }else if (partes[0].equals("sd")) {
                    int f2 = Integer.parseInt(partes[1].substring(1)); // Quita la "f" del segundo registro
                    int x3 = Integer.parseInt(partes[3].substring(1));
                    int inmediato = Integer.parseInt(partes[2]);
                    Instruccion ins = new Instruccion(4, f2, -1, x3, inmediato);
                    memoriaInstrucciones[i]=ins;
                    i++;
                }else if (partes[0].equals("fmult")) {
                    int f1 = Integer.parseInt(partes[1].substring(1)); // Quita la "f" del primer registro
                    int f2 = Integer.parseInt(partes[2].substring(1));
                    int f3 = Integer.parseInt(partes[3].substring(1));
                    int inmediato = 0;
                    Instruccion ins = new Instruccion(5, f1, f2, f3, inmediato);
                    memoriaInstrucciones[i]=ins;
                    i++;
                }
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return i;
    }

    public void inicializar_ER(EstacionReserva[][] ER) {
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < ER[a].length; b++) {
                EstacionReserva e = new EstacionReserva(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                ER[a][b] = e;
            }
        }


    }

    public void inicializar_banco_registros(Registro[] banco_registros, Instruccion[] mem) {
        int d = 2;
        for (int a = 0; a < banco_registros.length; a++) {
            Registro c = new Registro(d, 1, 0, -1);
            banco_registros[a] = c;
            //d += 1;
        }

    }


    public void mostrar_ER(EstacionReserva[][] ER) {
        // Implementa la lógica para mostrar el contenido de las estaciones de reserva.
        for (int a = 0; a < 3; a++) {
            for(int x = 0; x < ER[a].length; x++){
                if (ER[a][x].busy == 1) {
                    ER[a][x].pr();
                }
            }
        }
    }
    public void mostrar_ER2(EstacionReserva[][] ER) {
        // Implementa la lógica para mostrar el contenido de las estaciones de reserva.
        for (int a = 0; a < 3; a++) {
            for(int x = 0; x < ER[a].length; x++){
                ER[a][x].pr();

            }
        }
    }

    public void mostrar_banco_registros(Registro[] banco_registros) {
        // Implementa la lógica para mostrar el contenido del banco de registros.
        int i = 0;
        for (Registro bancoRegistro : banco_registros) {
            String a = bancoRegistro.pr();
            System.out.print("Registro " + i + ": " + a);
            i++;
        }

    }

    public void mostrar_UF(UnidadFuncional[] Funcional) {
        // Implementa la lógica para mostrar el contenido del banco de registros.
        for (UnidadFuncional UNIDAD : Funcional) {
            UNIDAD.pr();

        }

    }

    public void mostrar_ins(Instruccion[] memoria_instrucciones){
        int a = 0;
        for (Instruccion ins : memoria_instrucciones) {
            if(a < 3) {
                ins.pr();
                a+=1;
            }

        }
    }

    public void inicializar_UF(UnidadFuncional[] UF) {
        for (int a = 0; a < UF.length; a++) {
            UnidadFuncional uf = new UnidadFuncional(0, -1, -1, -1, -1, -1, -1, -1, -1);
            UF[a] = uf;
        }
    }



    public Procesador() {
        // Declaración de las variables que simulan la memoria de datos, de instrucciones y banco de registros.
        // Serán vectores
        Registro[] banco_registros = new Registro[REG]; // reg_t Registro del banco de registros
        /* Banco de registros */
        int[] memoria_datos = new int[DAT];
        /* Memoria de datos */
        Instruccion[] memoria_instrucciones1 = new Instruccion[INS];

        /* Memoria de instrucciones */
        UnidadFuncional[] UF = new UnidadFuncional[TOTAL_UF];
        EstacionReserva[][] ER = new EstacionReserva[TOTAL_UF][INS]; //ER_T
        int ins_fin = 0; /* total instrucciones que han finalizado la ejecución */
        int PC = 0; /* puntero a memoria de instrucciones */
        int[] p_er_cola = new int[TOTAL_UF]; /* puntero a memoria de instrucciones p_er_cola[0] apunta a la línea de ER[0] donde se almacenará la siguiente instrucción en esa ER */
        p_er_cola[0] = 0;
        p_er_cola[1] = 0;
        p_er_cola[2] = 0;
        // Inicialización del simulador
        int inst_prog = leer_programa(memoria_instrucciones1);/* total instrucciones programa */
        inicializar_ER(ER); //ER_T EstacionReserva
        inicializar_banco_registros(banco_registros, memoria_instrucciones1); // reg_t Registro del banco de registros
        inicializar_UF(UF);
        // inicializar_memoria_datos(memoria_datos); // ???????????

        int ciclo = 1; // ciclo de ejecución actual
        int[] s = new int[32];
        for(int a = 0; a < 32; a++){
            s[a]=a+10;
        }
        // Simulación. Bucle que se ejecuta mientras haya instrucciones en el ROB e instrucciones en la memoria de instrucciones
        ins_fin = inst_prog;
        while (ins_fin > 0 && ciclo < 23) {
            // En un ciclo de reloj se ejecutan las 5 etapas de procesamiento de una instrucción
            // Ejecutar cada una de las etapas.
            // Cada iteración del bucle simula las diferentes etapas de ejecución de la instrucción que se ejecutan en paralelo
            // En la etapa wb se libera una unidad funcional, otra operación puede comenzar en ese mismo ciclo.
            ins_fin = etapa_WB(ciclo, ins_fin, UF, ER, banco_registros,memoria_instrucciones1, s);
            etapa_EX(ciclo, UF, ER, p_er_cola,banco_registros,memoria_instrucciones1);
            if (PC < inst_prog) {
                etapa_ID_ISS(PC, ciclo, inst_prog, memoria_instrucciones1, ER, banco_registros, p_er_cola,s);
                PC += 1; // DENTRO  etapa_ID_ISS pero mejor aqui
            }
            //ins_fin -=1; // DENTRO  etapa_ WB
            // Mostrar el contenido de las distintas estructuras para ver cómo evoluciona la simulación
            System.out.println("Ciclo: " + ciclo);
            System.out.println("Mostrar Estacion Reserva");
            mostrar_ER(ER);
            System.out.println();
            System.out.println("Mostrar Banco Registros");
            mostrar_banco_registros(banco_registros);
            System.out.println();
            System.out.println("Mostrar UF");
            mostrar_UF(UF);
            System.out.println("ins_fin: " + ins_fin);
            System.out.println("Mostrar memoria[]");
            System.out.println(Arrays.toString(s));
            ciclo = ciclo + 1; // incrementamos contador de ciclo
            System.out.println("----------------------------------------------------------FIN CICLO----------------------------------------------------------");
        } // while
        //mostrar_ER2(ER);
    }

    public void etapa_ID_ISS(int PC, int ciclo, int inst_prog, Instruccion[] memoria_instrucciones, EstacionReserva[][] ER, Registro[] banco_registros, int[] p_er_cola, int[] rd) {
        // Implementa la lógica para la etapa ID/ISS.
        Instruccion ins0 = memoria_instrucciones[PC];
        int b = 0;
        int t = PC;
        if (ins0.cod == 1 || ins0.cod == 2) {// ALU suma resta
            b = p_er_cola[0];
            ER[0][b].busy = 1;
            ER[0][b].operacion = ins0.cod;
            if (banco_registros[ins0.rs].ok == 1) {
                ER[0][b].opa = banco_registros[ins0.rs].contenido;
                ER[0][b].opa_ok = 1;
                ER[0][b].clk_tick_ok_a = ciclo;
            } else {//banco_registros[ins0.rs].contenido == 0
                ER[0][b].opa = banco_registros[ins0.rs].TAG_ER;
                ER[0][b].opa_ok = 0;
                ER[0][b].clk_tick_ok_a = -1;
            }
            if (banco_registros[ins0.rt].ok == 1) {
                ER[0][b].opb = banco_registros[ins0.rt].contenido;
                ER[0][b].opb_ok = 1;
                ER[0][b].clk_tick_ok_b = ciclo;
            } else {//banco_registros[ins0.rs].contenido == 0
                ER[0][b].opb = banco_registros[ins0.rt].TAG_ER;
                ER[0][b].opb_ok = 0;
                ER[0][b].clk_tick_ok_b = -1;
            }
            ER[0][b].inmediato = ins0.inmediato;
            ER[0][b].TAG_ER = t;
            p_er_cola[0] += 1;
        } else if (ins0.cod == 3 || ins0.cod == 4) {// MEM LD Y SD
            b = p_er_cola[1];
            ER[1][b].busy = 1;
            ER[1][b].operacion = ins0.cod;
            if (ins0.cod == 3) {// MEM LD
                if (banco_registros[ins0.rt].ok == 1) { //ld ok? rt = opb, rs = opa
                    ER[1][b].opb = banco_registros[ins0.rt].contenido; //cambiar todos ?
                    ER[1][b].opb_ok = 1;
                    ER[1][b].clk_tick_ok_b = ciclo;
                    int mem = ER[1][b].opb + ins0.inmediato;
                    ER[1][b].opa = rd[mem];
                    ER[1][b].opa_ok = 1;
                    ER[1][b].clk_tick_ok_a = ciclo;
                } else {//banco_registros[ins0.rs].contenido == 0
                    ER[1][b].opb = banco_registros[ins0.rt].TAG_ER;
                    ER[1][b].opb_ok = 0;
                    ER[1][b].opa_ok = 0;
                    ER[1][b].clk_tick_ok_b = -1;
                    ER[1][b].clk_tick_ok_a = -1;
                }
                ER[1][b].inmediato = ins0.inmediato;
                ER[1][b].TAG_ER = t;
                p_er_cola[1] += 1;
            } else { // MEM SD
                if (banco_registros[ins0.rd].ok == 1) { //sd ok? rt = opb = r2 , rs = opa = -1, rd = r3,  si sd r3, 0(r2) // lo que quiero cargar ok falta donde huh????
                    ER[1][b].opa = banco_registros[ins0.rd].contenido;
                    ER[1][b].opa_ok = 1;
                    ER[1][b].clk_tick_ok_a = ciclo;
                    if (banco_registros[ins0.rt].ok == 1){
                        ER[1][b].opb=banco_registros[ins0.rt].contenido + ins0.inmediato;
                        ER[1][b].opb_ok=1;
                        ER[1][b].clk_tick_ok_b = ciclo;
                    }else{
                        ER[1][b].opb=banco_registros[ins0.rt].TAG_ER; // error
                        ER[1][b].opb_ok=0;
                        ER[1][b].clk_tick_ok_b = -1;
                    }
                }else{
                    ER[1][b].opa = banco_registros[ins0.rd].TAG_ER;// error
                    ER[1][b].opa_ok= 0;
                    ER[1][b].clk_tick_ok_a = -1;
                    if (banco_registros[ins0.rt].ok == 1){
                        ER[1][b].opb=banco_registros[ins0.rt].contenido + ins0.inmediato;
                        ER[1][b].opb_ok=1;
                        ER[1][b].clk_tick_ok_b = ciclo;
                    }else{
                        ER[1][b].opb=banco_registros[ins0.rt].TAG_ER;// error
                        ER[1][b].opb_ok=0;
                        ER[1][b].clk_tick_ok_b = -1;
                    }
                }
                ER[1][b].inmediato = ins0.inmediato;
                ER[1][b].TAG_ER = t;
                p_er_cola[1] += 1;
            }

        } else {// MULT
            b = p_er_cola[2];
            ER[2][b].busy = 1;
            ER[2][b].operacion = ins0.cod;
            if (banco_registros[ins0.rs].ok == 1) {
                ER[2][b].opa = banco_registros[ins0.rs].contenido;
                ER[2][b].opa_ok = 1;
                ER[2][b].clk_tick_ok_a = ciclo;
            } else {//banco_registros[ins0.rs].contenido == 0
                ER[2][b].opa = banco_registros[ins0.rs].TAG_ER;
                ER[2][b].opa_ok = 0;
                ER[2][b].clk_tick_ok_a = -1;
            }
            if (banco_registros[ins0.rt].ok == 1) {
                ER[2][b].opb = banco_registros[ins0.rt].contenido;
                ER[2][b].opb_ok = 1;
                ER[2][b].clk_tick_ok_b = ciclo;
            } else {//banco_registros[ins0.rs].contenido == 0
                ER[2][b].opb = banco_registros[ins0.rt].TAG_ER;
                ER[2][b].opb_ok = 0;
                ER[2][b].clk_tick_ok_b = -1;
            }
            ER[2][b].inmediato = ins0.inmediato;
            ER[2][b].TAG_ER = t;

            p_er_cola[2] += 1;
        }

        // 4..- Invalidar contenido del registro destino. Banco_registros[rd].ok = 0, Banco_registros[rd].TAG_ER = línea_TAG_ER, Banco_registros[rd].clk_tick_ok = ciclo

        if (ins0.cod != 4) {
            banco_registros[ins0.rd].TAG_ER = t;
            banco_registros[ins0.rd].ok = 0;
        }

        // falta para ins0.cod == 4 ?
    }

    public void etapa_EX(int ciclo, UnidadFuncional[] UF, EstacionReserva[][] ER, int[] p_er_cola, Registro[] banco_registros, Instruccion[] memoria_instrucciones) {
        // En todas las UF: Funcionamiento no segmentado. Hasta que no finaliza una instrucción completamente no empieza otra.
        // 1. En todas las UF que están en uso:
        // * Incremento un ciclo de operación.
        // *Si es el último: generar resultado y almacenarlo en UF[i].res, validarlo y actualizar ciclo
        // 2. Si alguna está libre: Enviar una nueva instrucción a ejecutar si hay instrucción disponible de ese tipo
        // * Se busca una instrucción que tenga los operandos disponibles en su ER
        // * Solo se puede enviar una instrucción
        // * Operaciones ALU se pueden enviar en cualquier orden:
        // SI los dos operandos están disponibles: Inicializa UF (operandos, contador de ciclos y línea ROB destino)
        // *load/store se enviarán en orden
        int s = -1;
        int i = 0; // contador unidades funcionales
        while (i < TOTAL_UF) { // revisa todas las UFs. Si está en uso, Incrementa ciclo. Si es el último, generar resultado y validarlo.
            UnidadFuncional uf_ = UF[i];  // puede que donde lo use tenga que llamar directamente de UF[i].loquesea
            // Establecer ciclos máximos para cada UF
            int max;
            int w = -1;
            if (i == ALU ) { // ALU = 0
                max = CICLOS_ALU;
            } else if (i == MEM && UF[i].uso == 1) { // MEM = 1
                max = CICLOS_MEM;

            } else { // i = MULT = 2
                max = CICLOS_MULT;

            }
            if (UF[i].uso == 1) {
                // si está en uso, se incrementa el ciclo y no se pueden enviar otra instrucción.

                if (UF[i].cont_ciclos < max) {

                    UF[i].cont_ciclos++; // incrementar el ciclo
                    if (UF[i].cont_ciclos == max) {
                        UF[i].res_ok = 1;// si se ha finalizado la operación validarlo
                        UF[i].clk_tick_ok = ciclo;// si se ha finalizado la operación actualizar ciclo
                        if (UF[i].operacion == 1) { //ADD  // Aquí deberías completar la lógica para generar el resultado
                            UF[i].res = UF[i].opa + UF[i].opb;
                        } else if (UF[i].operacion == 2) { //SUB
                            UF[i].res = UF[i].opa - UF[i].opb;
                        } else if (UF[i].operacion == 3) { //LD
                            UF[i].res = UF[i].opa;// huh?
                        } else if (UF[i].operacion == 4) { //SD
                            UF[i].res = UF[i].opa; // huh?
                        } else if (UF[i].operacion == 5) { //MULT
                            UF[i].res = UF[i].opa * UF[i].opb;
                        }
                    } // ciclo max
                } // en uso
            }
            i++;
        } // Fin while incremento contador ciclos de operación de todas las UF en funcinamiento

        // Enviar una instrucción si está disponible en aquella unidad que esté libre. Orden para enviar: ADD, MULT, STORE y LOAD
        int enviar = 0; // variable para controlar el envio.
        EstacionReserva[] er_; // variable para la estación de reserva a considerar
        UnidadFuncional uf_; // variable para la unidad funcional a considerar

        // Unidad funcional ADD SUB (ALU)
        if (UF[ALU].uso == 0) { // si está libre, buscar instrucción para enviar
            er_ = ER[ALU];
            int j = 0; // contador de líneas de ER[0] desde 0 hasta fin
            int fin = p_er_cola[0]; // última línea insertada
            while (j < fin && enviar == 0) { // se puede enviar cualquier instrucción que tenga los operandos disponibles
                if (ER[ALU][j].busy == 1) { // comprueba si los operandos están disponibles para operación ALU
                    if(ER[ALU][j].opa_ok == 1 && ER[ALU][j].clk_tick_ok_a < ciclo &&
                            ER[ALU][j].opb_ok == 1 && ER[ALU][j].clk_tick_ok_b < ciclo) { // operandos disponibles
                        UF[ALU].operacion = ER[ALU][j].operacion;
                        UF[ALU].opb = ER[ALU][j].opb;
                        UF[ALU].opa = ER[ALU][j].opa;
                        UF[ALU].TAG_ER = ER[ALU][j].TAG_ER;
                        UF[ALU].res = 0;
                        UF[ALU].res_ok = 0;
                        UF[ALU].uso = 1;
                        UF[ALU].clk_tick_ok = 0;
                        UF[ALU].cont_ciclos = 1;
                        // Enviar operación a ejecutar a UF actualizando UF[i] correspondiente;
                        // enviar = 1 indicando que ya no se pueden enviar a ejecutar más instrucciones
                        enviar = 1;
                        // Aquí deberías completar la lógica para enviar la operación a la UF correspondiente
                    }
                }
                j++;
            }
        }

        // Unidad funcional MULT
        if (UF[MULT].uso == 0 && enviar == 0) { // si está libre y no se ha enviado ninguna anterior. Solo se envia la primera// buscar instrucción a ejecutar en todas las líneas validas de er_
            er_ = ER[MULT]; // ALU mult
            uf_ = UF[MULT];
            int fin = p_er_cola[2]; // última línea insertada
            int j = 0; // contador de líneas de ER[2] desde 0 hasta fin
            while (enviar == 0 && j < fin) {
                // búsqueda de instrucción a ejecutar en todas las líneas validas de er_
                if (er_[j].busy == 1) { // comprueba si los operandos están disponibles para operación ALU
                    if (ER[MULT][j].opa_ok == 1 && ER[MULT][j].clk_tick_ok_a < ciclo &&
                            ER[MULT][j].opb_ok == 1 && ER[MULT][j].clk_tick_ok_b < ciclo) { // operandos disponibles creo que esta bien ya que parecido a ALU
                        UF[MULT].operacion = ER[MULT][j].operacion;
                        UF[MULT].opb = ER[MULT][j].opb;
                        UF[MULT].opa = ER[MULT][j].opa;
                        UF[MULT].TAG_ER = ER[MULT][j].TAG_ER;
                        UF[MULT].res = 0;
                        UF[MULT].res_ok = 0;
                        UF[MULT].uso = 1;
                        UF[MULT].clk_tick_ok = 0;
                        UF[MULT].cont_ciclos = 1;
                        // Enviar operación a ejecutar a UF actualizando UF[i] correspondiente;
                        // enviar = 1 indicando que ya no se pueden enviar a ejecutar más instrucciones
                        enviar = 1;
                        // Aquí deberías completar la lógica para enviar la operación a la UF correspondiente
                    }
                }
                j++;
            }
        }

        // Unidad funcional MEM
        if (UF[1].uso == 0 && enviar == 0) { // si UF está libre y no se ha enviado ninguna anterior. Solo se envia la primera
            er_ = ER[MEM]; // MEM
            uf_ = UF[MEM];
            // buscar primera instrucción en ER (han ido acabando instrucciones busy =0. Buscar primera línea con busy a 1)
            i = 0;
            while (er_[i].busy == 0 && i < p_er_cola[MEM]) i++;
            if (i < p_er_cola[MEM]) {
                if (er_[i].operacion == sd && er_[i].opa_ok == 1 && er_[i].clk_tick_ok_a < ciclo &&
                        er_[i].opb_ok == 1 && er_[i].clk_tick_ok_b < ciclo ) { // operandos disponibles para store
                    // Calcular dir de memoria = er_[i].opa + er_[i].inmediato)
                    // operación de escritura en memoria inicializando UF
                    UF[MEM].cont_ciclos = 1;
                    UF[MEM].uso = 1;
                    UF[MEM].res_ok = 0;
                    UF[MEM].operacion = sd;
                    UF[MEM].TAG_ER = er_[i].TAG_ER; // huh??
                    UF[MEM].opb = er_[i].opb;
                    UF[MEM].opa = er_[i].opa;
                    UF[MEM].res=0;

                    // Aquí deberías completar la lógica para enviar la operación a la UF correspondiente
                } else if (er_[i].operacion == ld && er_[i].opa_ok == 1 && er_[i].clk_tick_ok_a < ciclo &&
                        er_[i].opb_ok == 1 && er_[i].clk_tick_ok_b < ciclo) { // load.
                    // Calcular dir de memoria = er_[i].opa + er_[i].inmediato) no se si necesario
                    int dir_mem = er_[i].opa + er_[i].inmediato;
                    // operación de lectura en memoria inicializando UF

                    UF[MEM].cont_ciclos = 1;
                    UF[MEM].uso = 1;
                    UF[MEM].res_ok = 0;
                    UF[MEM].operacion = ld;
                    UF[MEM].TAG_ER = er_[i].TAG_ER;
                    UF[MEM].opb = er_[i].opb;
                    UF[MEM].opa = er_[i].opa;
                    UF[MEM].res=0;

                    // Aquí deberías completar la lógica para enviar la operación a la UF correspondiente
                }
            }
        }
    }

    public int etapa_WB(int ciclo, int ins_fin, UnidadFuncional[] UF, EstacionReserva[][] ER, Registro[] banco_registros, Instruccion[] memoria_instrucciones, int[] sr) {
        // Objetivo:
        // 1. Se busca el primer resultado válido en una unidad funcional
        //    resultado válido: UF[i].res_ok = 1 y UF[i].clk_tick_ok < ciclo_actual
        // Acciones:
        //    Se almacena el resultado en el registro y se actualizan los operandos que esperaban ese resultado en las estaciones de reserva
        //    Se deja libre UF. Se pone todo a 0
        //    Si no hay ningún resultado disponible, no se hace nada
        int i = 0; // contador de unidades funcionales. Hay que comprobar en todas ellas.
        int bucle = 0; // controla salir del bucle cuando encuentra un resultado valido. Solo WB un resultado
        while (bucle == 0 && i < TOTAL_UF) { // busca resultado valido en todas las UF. El primero que encuentra ejecuta WB
            if (UF[i].uso == 1 && UF[i].res_ok == 1 && UF[i].clk_tick_ok < ciclo) {
                int x = UF[i].TAG_ER; // x es la etiqueta de la línea de estación de reserva donde está almacenada instrucción que espera ese resultado
                int res = UF[i].res; // resultado generado

                // Cualquier registro y que cumpla: banco_registros[y].TAG_ER == x entonces banco_registros[y].contenido = res (ok y ciclo)
                //Si no lo hay no se hace nada. Instrucción store o un registro sobreescrito por otra instrucción posterior.
                //Dejar libre UF. Poner todo a 0s
                //Limpiar ER donde estaba esa instrucción almacenada. Línea etiquetada por x. Buscar esa línea: ER[i].TAG_ER == x
                ins_fin = ins_fin - 1; //instrucción finalizada
                int r = 0;
                // Actualizar registros
                for (int y = 0; y < REG; y++) {
                    if (banco_registros[y].TAG_ER == x && r == 0 && UF[i].operacion != 4) {
                        banco_registros[y].contenido = res;
                        banco_registros[y].ok = 1;
                        banco_registros[y].clk_tick_ok = ciclo;
                        r = 1;
                    }

                }
                if (UF[i].operacion == 4) {

                    sr[UF[i].opb] = res;
                }
                for (int a = 0; a < TOTAL_UF; a++) {//OPAA esto hay que hacerlo pero no se si esta bien
                    for (int b = 0; b < ER[a].length; b++) {//OPAA
                        if (ER[a][b].opb == x && ER[a][b].opb_ok == 0) {
                            ER[a][b].opb = res;
                            ER[a][b].opb_ok = 1;
                            ER[a][b].clk_tick_ok_b = ciclo;
                        }
                        if (a == MEM) {
                            if (ER[a][b].operacion==ld){
                                if (ER[a][b].opb_ok == 1 && ER[a][b].opa == x && ER[a][b].opa_ok == 0) {
                                    ER[a][b].opa = sr[res+ER[a][b].inmediato];
                                    ER[a][b].opa_ok = 1;
                                    ER[a][b].clk_tick_ok_a = ciclo;
                                }
                            }else {
                                if (ER[a][b].opa == x && ER[a][b].opa_ok == 0) {
                                    ER[a][b].opa = res;
                                    ER[a][b].opa_ok = 1;
                                    ER[a][b].clk_tick_ok_a = ciclo;
                                }
                            }
                        } else {
                            if (ER[a][b].opa == x && ER[a][b].opa_ok == 0) {
                                ER[a][b].opa = res;
                                ER[a][b].opa_ok = 1;
                                ER[a][b].clk_tick_ok_a = ciclo;
                            }
                        }
                    }
                    // Limpiar ER donde estaba esa instrucción almacenada. Línea etiquetada por x.
                    int ok = 0;
                    for (int k = 0; k < TOTAL_UF; k++) {
                        for (int j = 0; j < INS; j++) {
                            if (ER[k][j].TAG_ER == x) {
                                ER[k][j].busy = 0;
                                ER[k][j].operacion = -1;
                                ER[k][j].opa = -1;
                                ER[k][j].opa_ok = -1;
                                ER[k][j].clk_tick_ok_a = -1;
                                ER[k][j].opb = -1;
                                ER[k][j].opb_ok = -1;
                                ER[k][j].clk_tick_ok_b = -1;
                                ER[k][j].inmediato = -1;
                                ER[k][j].TAG_ER = -1;
                                ok = 1;
                            }
                        }
                    }

                    // Dejar libre UF. Poner todo a 0s
                    UF[i].uso = 0;
                    UF[i].cont_ciclos = -1;
                    UF[i].TAG_ER = -1;
                    UF[i].opa = -1;
                    UF[i].opb = -1;
                    UF[i].operacion = -1;
                    UF[i].res = -1;
                    UF[i].res_ok = -1;
                    UF[i].clk_tick_ok = -1;
                    bucle = 1; // salir del bucle no habrá más iteraciones del bucle
                }

            }
            i++;

        }
        return ins_fin;
    }
}