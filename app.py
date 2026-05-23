from flask import Flask, render_template, url_for, request
import numpy as np
import numpy_financial as npf
import math

app = Flask(__name__)

# ──────────────────────────────────────────────
# Rate conversion utility
# ──────────────────────────────────────────────
PERIODS_PER_YEAR = {
    'Diario': 360,
    'Semanal': 52,
    'Quincenal': 24,
    'Mensual': 12,
    'Bimestral': 6,
    'Trimestral': 4,
    'Cuatrimestral': 3,
    'Semestral': 2,
    'Anual': 1,
}

UNIDAD_N_PERIODS = {
    'Días': 360,
    'Meses': 12,
    'Años': 1,
}


def convert_rate(rate_pct, rate_type, payment_frequency):
    """
    Convert a given rate (in percentage, e.g. 12 for 12%) to an effective
    periodic rate (decimal) that matches the payment frequency.
    """
    r = rate_pct / 100.0  # convert % to decimal
    ppy = PERIODS_PER_YEAR[payment_frequency]

    # Step 1: convert to Effective Annual Rate (EA)
    if rate_type == 'EA':
        ea = r
    elif rate_type == 'EM':
        ea = (1 + r) ** 12 - 1
    elif rate_type == 'ET':
        ea = (1 + r) ** 4 - 1
    elif rate_type == 'ES':
        # Efectiva Semestral → EA
        ea = (1 + r) ** 2 - 1
    elif rate_type == 'NA':
        # Nominal Annual (commonly compounded monthly)
        ea = (1 + r / 12) ** 12 - 1 
    # --- Tasas Nominales con capitalización FIJA ---
    elif rate_type in ('MV', 'CM', 'NM'):
        # Capitalización MENSUAL (m=12) — sin importar ppy
        tasa_mensual = r / 12
        ea = (1 + tasa_mensual) ** 12 - 1
    elif rate_type in ('TV', 'CT', 'NT'):
        # Capitalización TRIMESTRAL (m=4) — sin importar ppy
        tasa_trimestral = r / 4
        ea = (1 + tasa_trimestral) ** 4 - 1
    elif rate_type in ('SV', 'CS', 'NS'):
        # Capitalización SEMESTRAL (m=2) — sin importar ppy
        tasa_semestral = r / 2
        ea = (1 + tasa_semestral) ** 2 - 1
    elif rate_type == 'NV':
    # Nominal Vencida genérica: la frecuencia de capitalización
    # es igual a la frecuencia de pago (ppy)
    # Cubre frecuencias no estándar: bimestral, quincenal, semanal, diaria, etc.
        tasa_periodica_vencida = r / ppy
        ea = (1 + tasa_periodica_vencida) ** ppy - 1
    elif rate_type == 'NAA':
        # Nominal Annual Anticipada, compounded monthly
        # monthly anticipada = NAA / 12
        # monthly vencida = monthly_anticipada / (1 - monthly_anticipada)
        tasa_periodica_anticipada = r / ppy
        if tasa_periodica_anticipada >= 1:
            raise ValueError(
                f"La tasa NAA es demasiado alta: " 
                f"tasa periódica anticipada = {tasa_periodica_anticipada:.4%} >= 100.")
        tasa_periodica_vencida = tasa_periodica_anticipada / (1 - tasa_periodica_anticipada)
        ea = (1 + tasa_periodica_vencida) ** ppy - 1
    else:
        ea = r  # fallback

    # Step 2: convert EA → effective periodic rate
    periodic_rate = (1 + ea) ** (1 / ppy) - 1
    return periodic_rate


def ea_to_rate_type(ea, target_type, ppy):
    """
    Convertir una Tasa Efectiva Anual (decimal) al tipo de tasa solicitado.
    Retorna un decimal (ej. 0.12 para 12%).
    """
    if target_type == 'EA':
        return ea
    elif target_type == 'EM':
        return (1 + ea) ** (1 / 12) - 1
    elif target_type == 'ET':
        return (1 + ea) ** (1 / 4) - 1
    elif target_type == 'ES':
        return (1 + ea) ** (1 / 2) - 1
    elif target_type in ('MV', 'CM', 'NM'):
        tasa_mensual = (1 + ea) ** (1 / 12) - 1
        return tasa_mensual * 12
    elif target_type in ('TV', 'CT', 'NT'):
        tasa_trimestral = (1 + ea) ** (1 / 4) - 1
        return tasa_trimestral * 4
    elif target_type in ('SV', 'CS', 'NS'):
        tasa_semestral = (1 + ea) ** (1 / 2) - 1
        return tasa_semestral * 2
    elif target_type == 'NV':
        tasa_periodica = (1 + ea) ** (1 / ppy) - 1
        return tasa_periodica * ppy
    elif target_type == 'NAA':
        tasa_periodica_vencida = (1 + ea) ** (1 / ppy) - 1
        if tasa_periodica_vencida >= 1:
            raise ValueError(
                f"La tasa EA es demasiado alta para convertir a NAA: "
                f"tasa periódica vencida = {tasa_periodica_vencida:.4%} >= 100%")
        tasa_periodica_anticipada = tasa_periodica_vencida / (1 + tasa_periodica_vencida)
        return tasa_periodica_anticipada * ppy
    else:
        return ea  # fallback



def calculate_anualidad(tipo, desconocida, vp, vf, r, i_rate, n, m):
    """
    Return (result, error_string).
    All monetary values are floats (positive in, positive out convention).
    i_rate is the effective periodic rate (decimal).
    """
    when = 'end' if tipo == 'vencida' or tipo == 'perpetua' else 'begin'

    try:
        # ── Valor Presente ──
        if desconocida == 'vp':
            if tipo == 'perpetua':
                if when == 'end':
                    result = r / i_rate
                else:
                    result = r / i_rate * (1 + i_rate)
                return result, None
            elif tipo == 'diferida':
                # VP = R * [1-(1+i)^-n] / i * (1+i)^-m
                factor = (1 - (1 + i_rate) ** (-n)) / i_rate
                if when == 'begin':
                    factor *= (1 + i_rate)
                factor *= (1 + i_rate) ** (-m)
                result = r * factor
                return result, None
            else:
                # Ordinary or annuity due
                result = npf.pv(i_rate, n, -r, fv=vf if vf else 0, when=when)
                return result, None

        # ── Valor Futuro ──
        elif desconocida == 'vf':
            if tipo == 'perpetua':
                return None, "El Valor Futuro en una anualidad perpetua es infinito / no está definido."
            else:
                result = npf.fv(i_rate, n, -r, pv=vp if vp else 0, when=when)
                return result, None

        # ── Renta / Cuota ──
        elif desconocida == 'r':
            if tipo == 'perpetua':
                result = vp * i_rate
                return result, None
            elif tipo == 'diferida':
                # VP = R * [1-(1+i)^-n] / i * (1+i)^-m
                factor = (1 - (1 + i_rate) ** (-n)) / i_rate
                if when == 'begin':
                    factor *= (1 + i_rate)
                factor *= (1 + i_rate) ** (-m)
                result = vp / factor
                return result, None
            else:
                result = npf.pmt(i_rate, n, vp if vp else 0, fv=vf if vf else 0, when=when)
                return abs(result), None

        # ── Tasa de interés ──
        elif desconocida == 'i':
            if tipo == 'perpetua':
                result = r / vp
                return result, None
            else:
                result = npf.rate(n, -r, vp if vp else 0, fv=vf if vf else 0, when=when, guess=0.1)
                return result, None

        # ── Número de períodos ──
        elif desconocida == 'n':
            if tipo == 'perpetua':
                return None, "El número de períodos en una anualidad perpetua es infinito."
            else:
                result = npf.nper(i_rate, -r, vp if vp else 0, fv=vf if vf else 0, when=when)
                return result, None

        # ── Tiempo de diferimiento ──
        elif desconocida == 'm':
            if tipo == 'perpetua':
                return None, "El diferimiento no aplica para anualidades perpetuas."
            if tipo != 'diferida':
                return None, "El diferimiento (m) solo aplica para anualidades diferidas."
            # VP = R * [1-(1+i)^-n] / i * (1+i)^-m * (factor anticipada)
            # Solve for m:
            # (1+i)^-m = VP / (R * [1-(1+i)^-n]/i * factor_anticipada)
            factor = (1 - (1 + i_rate) ** (-n)) / i_rate
            if when == 'begin':
                factor *= (1 + i_rate)
            base = vp / (r * factor)
            if base <= 0:
                return None, "No se puede calcular m con los valores ingresados (razón no positiva)."
            m_calc = -math.log(base) / math.log(1 + i_rate)
            return m_calc, None

        else:
            return None, "Variable desconocida no válida."

    except Exception as e:
        return None, f"Error en el cálculo: {str(e)}"



def calculate_interes_compuesto(desconocida, vp, vf, periodic_rate, n,
                                 unidad_tiempo_n='Meses', tipo_tasa='EA', ppy=12):
    """
    Return (result, error_string) for compound interest calculations.
    periodic_rate is the effective periodic rate (decimal).
    """
    try:
        if desconocida == 'vf':
            # VF = VP * (1 + i)^n
            result = vp * (1 + periodic_rate) ** n
            return result, None
        elif desconocida == 'vp':
            # VP = VF / (1 + i)^n
            result = vf / (1 + periodic_rate) ** n
            return result, None
        elif desconocida == 'n':
            # n = log(VF/VP) / log(1 + i)
            if vp <= 0 or vf <= 0:
                return None, "VP y VF deben ser valores positivos para calcular n."
            result = math.log(vf / vp) / math.log(1 + periodic_rate)
            return result, None
        elif desconocida == 'i':
            # i = (VF/VP)^(1/n) - 1  (tasa efectiva del período base n)
            if vp <= 0 or vf <= 0 or n <= 0:
                return None, "VP, VF y n deben ser valores positivos para calcular i."
            raw_i = (vf / vp) ** (1 / n) - 1
            # Convertir la tasa cruda a EA según la unidad temporal de n
            n_ppy = UNIDAD_N_PERIODS.get(unidad_tiempo_n, 12)
            if n_ppy == 1:
                ea = raw_i
            else:
                ea = (1 + raw_i) ** n_ppy - 1
            # Convertir EA al tipo de tasa solicitado por el usuario
            result = ea_to_rate_type(ea, tipo_tasa, ppy)
            return result, None
        else:
            return None, "Variable desconocida no válida."
    except Exception as e:
        return None, f"Error en el cálculo: {str(e)}"

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/amortizacion', methods=["GET", "POST"])
def amortizacion():
    tabla = None
    data = {}
    if request.method == "POST":
        capital = float(request.form["capital"])
        tasa_val = float(request.form["tasa_interes"])
        tipo_tasa = request.form.get("tipo_tasa", "EA")
        frecuencia_pago = request.form.get("frecuencia_pago", "Mensual")
        periodos = int(request.form["periodos"])

        # Optional extra payment
        raw_periodo_extra = request.form.get("periodo_extraordinario", "").strip()
        raw_monto_extra = request.form.get("monto_extraordinario", "").strip()

        # Convert rate
        periodic_rate = convert_rate(tasa_val, tipo_tasa, frecuencia_pago)

        # Calculate original fixed cuota (French system)
        cuota_original = npf.pmt(periodic_rate, periodos, -capital)

        # Build amortization table
        tabla = []

        # Period 0: show only the initial balance
        tabla.append({
            'periodo': 0,
            'saldo': round(capital, 3),
            'interes': None,
            'cuota': None,
            'amortizacion': None,
            'extra': 0.0
        })

        saldo = capital

        tiene_extra = raw_periodo_extra != "" and raw_monto_extra != ""
        periodo_extra_int = int(raw_periodo_extra) if tiene_extra else None
        monto_extra_float = float(raw_monto_extra) if tiene_extra else 0

        cuota_actual = cuota_original

        for i in range(1, periodos + 1):
            interes = saldo * periodic_rate
            amortizacion = cuota_actual - interes

            # Last-period adjustment to avoid negative balance
            if amortizacion > saldo:
                amortizacion = saldo
                cuota_actual = amortizacion + interes

            saldo_nuevo = saldo - amortizacion

            fila = {
                'periodo': i,
                'saldo': round(saldo_nuevo, 3),
                'interes': round(interes, 3),
                'cuota': round(cuota_actual, 3),
                'amortizacion': round(amortizacion, 3),
                'extra': 0.0
            }

            # Apply extra payment at the specified period
            if tiene_extra and i == periodo_extra_int:
                fila['extra'] = round(monto_extra_float, 3)
                saldo_nuevo -= monto_extra_float
                fila['saldo'] = round(saldo_nuevo, 3)

                # Recalculate cuota with remaining balance and remaining periods
                periodos_restantes = periodos - i
                if periodos_restantes > 0 and saldo_nuevo > 0.01:
                    cuota_actual = npf.pmt(periodic_rate, periodos_restantes, -saldo_nuevo)
                elif saldo_nuevo <= 0.01:
                    tabla.append(fila)
                    break

            tabla.append(fila)
            saldo = max(saldo_nuevo, 0.0)

            if saldo <= 0.001:
                break

        data['capital'] = capital
        data['tasa_interes'] = tasa_val
        data['tipo_tasa'] = tipo_tasa
        data['frecuencia_pago'] = frecuencia_pago
        data['periodos'] = periodos
        data['cuota_original'] = round(cuota_original, 3)
        data['tiene_extra'] = tiene_extra
        if tiene_extra:
            data['periodo_extra'] = periodo_extra_int
            data['monto_extra'] = round(monto_extra_float, 3)

    return render_template('amortizacion.html', tabla=tabla, data=data)


@app.route('/anualidad', methods=["GET", "POST"])
def anualidad():
    resultado = None
    error = None
    data = {}

    if request.method == "POST":
        # ── Read form ──
        data['tipo_anualidad'] = request.form.get('tipo_anualidad', 'vencida')
        data['variable_desconocida'] = request.form.get('variable_desconocida', 'vp')
        data['tipo_tasa'] = request.form.get('tipo_tasa', 'EA')
        data['frecuencia_pago'] = request.form.get('frecuencia_pago', 'Mensual')

        # Raw values (may be empty)
        raw_vp = request.form.get('vp', '').strip()
        raw_vf = request.form.get('vf', '').strip()
        raw_r = request.form.get('renta', '').strip()
        raw_i = request.form.get('tasa_interes', '').strip()
        raw_n = request.form.get('periodos', '').strip()
        raw_m = request.form.get('tiempo_diferimiento', '').strip()

        vp = float(raw_vp) if raw_vp else 0.0
        vf = float(raw_vf) if raw_vf else 0.0
        r = float(raw_r) if raw_r else 0.0
        tasa_val = float(raw_i) if raw_i else 0.0
        n = float(raw_n) if raw_n else 0.0
        m = float(raw_m) if raw_m else 0.0

        data['vp'] = vp
        data['vf'] = vf
        data['renta'] = r
        data['tasa_interes'] = tasa_val
        data['periodos'] = n
        data['tiempo_diferimiento'] = m

        # ── Validate required inputs ──
        desconocida = data['variable_desconocida']
        tipo = data['tipo_anualidad']

        if desconocida == 'vp' and r == 0:
            error = "Debes ingresar un valor para la Renta (R) para calcular VP."
        elif desconocida == 'vf' and r == 0:
            error = "Debes ingresar un valor para la Renta (R) para calcular VF."
        elif desconocida == 'r' and vp == 0 and vf == 0:
            error = "Debes ingresar al menos VP o VF para calcular la Renta (R)."
        elif desconocida == 'i' and (vp == 0 or r == 0):
            error = "Debes ingresar VP y Renta (R) para calcular la tasa de interés."
        elif desconocida == 'n' and (vp == 0 or r == 0):
            error = "Debes ingresar VP y Renta (R) para calcular el número de períodos."
        elif desconocida == 'm' and (vp == 0 or r == 0 or n == 0):
            error = "Debes ingresar VP, Renta (R) y n para calcular el diferimiento (m)."
        elif tipo == 'perpetua' and desconocida == 'vf':
            error = "El Valor Futuro en una anualidad perpetua es infinito / no está definido."
        elif tipo == 'perpetua' and desconocida == 'n':
            error = "El número de períodos en una anualidad perpetua es infinito."
        elif tipo == 'perpetua' and desconocida == 'm':
            error = "El diferimiento no aplica para anualidades perpetuas."

        # ── Rate conversion ──
        if error is None:
            try:
                periodic_rate = convert_rate(tasa_val, data['tipo_tasa'], data['frecuencia_pago'])
            except ValueError as ve:
                error = str(ve)
            except Exception:
                error = "Error al convertir la tasa de interés. Verifica el valor y tipo de tasa."

        # ── Calculate ──
        if error is None:
            resultado, error = calculate_anualidad(
                tipo, desconocida,
                vp if desconocida != 'vp' else 0,
                vf if desconocida != 'vf' else 0,
                r if desconocida != 'r' else 0,
                periodic_rate,
                n if desconocida != 'n' else 0,
                m if desconocida != 'm' else 0,
            )

            if resultado is not None:
                # Round monetary values
                if desconocida in ('vp', 'vf', 'r'):
                    resultado = round(resultado, 3)
                elif desconocida in ('n', 'm'):
                    resultado = np.round(resultado, 4)
                elif desconocida == 'i':
                    resultado = round(resultado, 8)

    return render_template('anualidad.html', resultado=resultado, error=error, data=data)


@app.route('/interes-compuesto', methods=["GET", "POST"])
def interes_compuesto():
    resultado = None
    error = None
    data = {}

    if request.method == "POST":
        data['variable_desconocida'] = request.form.get('variable_desconocida', 'vf')
        data['tipo_tasa'] = request.form.get('tipo_tasa', 'EA')
        data['frecuencia_pago'] = request.form.get('frecuencia_pago', 'Mensual')
        data['unidad_tiempo_n'] = request.form.get('unidad_tiempo_n', 'Meses')

        raw_vp = request.form.get('vp', '').strip()
        raw_vf = request.form.get('vf', '').strip()
        raw_i = request.form.get('tasa_interes', '').strip()
        raw_n_years = request.form.get('n_years', '0').strip()
        raw_n_months = request.form.get('n_months', '0').strip()
        raw_n_days = request.form.get('n_days', '0').strip()
        raw_unidad = request.form.get('unidad_tiempo_n', 'Meses')

        vp = float(raw_vp) if raw_vp else 0.0
        vf = float(raw_vf) if raw_vf else 0.0
        tasa_val = float(raw_i) if raw_i else 0.0
        n_years = float(raw_n_years) if raw_n_years else 0.0
        n_months = float(raw_n_months) if raw_n_months else 0.0
        n_days = float(raw_n_days) if raw_n_days else 0.0
        unidad_tiempo_n_val = data.get('unidad_tiempo_n', 'Meses')
        # Compute total n from años/meses/días based on unidad_tiempo_n
        if unidad_tiempo_n_val == 'Años':
            n = n_years + n_months / 12.0 + n_days / 360.0
        elif unidad_tiempo_n_val == 'Meses':
            n = n_years * 12 + n_months + n_days / 30.0
        elif unidad_tiempo_n_val == 'Días':
            n = n_years * 360 + n_months * 30 + n_days
        else:
            n = n_years + n_months / 12.0 + n_days / 360.0

        # ── Validate non-negative values ──
        if vp < 0:
            error = "El Valor Presente (VP) no puede ser negativo."
        elif vf < 0:
            error = "El Valor Futuro (VF) no puede ser negativo."
        elif tasa_val < 0:
            error = "La tasa de interés no puede ser negativa."
        elif n_years < 0 or n_months < 0 or n_days < 0:
            error = "Los años, meses y días no pueden ser negativos."

        data['vp'] = vp
        data['vf'] = vf
        data['tasa_interes'] = tasa_val
        data['n_years'] = n_years
        data['n_months'] = n_months
        data['n_days'] = n_days
        data['periodos'] = n
        data['unidad_tiempo_n'] = raw_unidad

        desconocida = data['variable_desconocida']

        # ── Validate required inputs ──
        if desconocida == 'vf' and (vp == 0 or tasa_val == 0 or n == 0):
            error = "Debes ingresar VP, tasa de interés y número de períodos para calcular VF."
        elif desconocida == 'vp' and (vf == 0 or tasa_val == 0 or n == 0):
            error = "Debes ingresar VF, tasa de interés y número de períodos para calcular VP."
        elif desconocida == 'n' and (vp == 0 or vf == 0 or tasa_val == 0):
            error = "Debes ingresar VP, VF y tasa de interés para calcular n."
        elif desconocida == 'i' and (vp == 0 or vf == 0 or n == 0):
            error = "Debes ingresar VP, VF y número de períodos para calcular i."

        # ── Rate conversion ──
        if error is None:
            try:
                periodic_rate = convert_rate(tasa_val, data['tipo_tasa'], data['frecuencia_pago'])
            except ValueError as ve:
                error = str(ve)
            except Exception:
                error = "Error al convertir la tasa de interés. Verifica el valor y tipo de tasa."

        # ── Calculate ──
        if error is None:
            resultado, error = calculate_interes_compuesto(
                desconocida,
                vp if desconocida != 'vp' else 0,
                vf if desconocida != 'vf' else 0,
                periodic_rate,
                n if desconocida != 'n' else 0,
                unidad_tiempo_n=data.get('unidad_tiempo_n', 'Meses'),
                tipo_tasa=data['tipo_tasa'],
                ppy=PERIODS_PER_YEAR[data['frecuencia_pago']],
            )

            if resultado is not None:
                if desconocida in ('vp', 'vf'):
                    resultado = round(resultado, 3)
                elif desconocida == 'i':
                    resultado = round(resultado, 8)
                elif desconocida == 'n':
                    resultado = np.round(resultado, 4)
                # Build n breakdown for result display
                if desconocida == 'n' and resultado is not None:
                    if data['unidad_tiempo_n'] == 'Años':
                        yrs = int(resultado)
                        mths = int(round((resultado - yrs) * 12))
                        data['n_breakdown'] = f"{yrs} años, {mths} meses"
                    elif data['unidad_tiempo_n'] == 'Meses':
                        yrs = int(resultado // 12)
                        mths = int(resultado % 12)
                        data['n_breakdown'] = f"{yrs} años, {mths} meses"

    return render_template('interes_compuesto.html', resultado=resultado, error=error, data=data)


@app.route('/acerca-de')
def acerca_de():
    return render_template('acerca_de.html')


if __name__ == "__main__":
    app.run(debug=True)