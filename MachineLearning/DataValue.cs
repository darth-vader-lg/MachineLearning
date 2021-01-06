using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Valore di vista dati
   /// </summary>
   public struct DataValue
   {
      #region Fields
      /// <summary>
      /// Dizionario di conversioni
      /// </summary>
      private static readonly Dictionary<(Type from, Type to), Func<object, object>> Converter = new Dictionary<(Type from, Type to), Func<object, object>>
      {
         { (typeof(VBuffer<sbyte>), typeof(sbyte[])), new Func<object, object>(value => VBufferToArray<sbyte>(value)) },
         { (typeof(VBuffer<byte>), typeof(byte[])), new Func<object, object>(value => VBufferToArray<byte>(value)) },
         { (typeof(VBuffer<short>), typeof(short[])), new Func<object, object>(value => VBufferToArray<short>(value)) },
         { (typeof(VBuffer<ushort>), typeof(ushort[])), new Func<object, object>(value => VBufferToArray<ushort>(value)) },
         { (typeof(VBuffer<int>), typeof(int[])), new Func<object, object>(value => VBufferToArray<int>(value)) },
         { (typeof(VBuffer<uint>), typeof(uint[])), new Func<object, object>(value => VBufferToArray<uint>(value)) },
         { (typeof(VBuffer<long>), typeof(long[])), new Func<object, object>(value => VBufferToArray<long>(value)) },
         { (typeof(VBuffer<ulong>), typeof(ulong[])), new Func<object, object>(value => VBufferToArray<ulong>(value)) },
         { (typeof(VBuffer<float>), typeof(float[])), new Func<object, object>(value => VBufferToArray<float>(value)) },
         { (typeof(VBuffer<double>), typeof(double[])), new Func<object, object>(value => VBufferToArray<double>(value)) },
         { (typeof(ReadOnlyMemory<char>), typeof(string)), new Func<object, object>(value => ((ReadOnlyMemory<char>)value).ToString()) },
         { (typeof(VBuffer<ReadOnlyMemory<char>>), typeof(string[])), new Func<object, object>(value => ((VBuffer<ReadOnlyMemory<char>>)value).DenseValues().Select(s => s.ToString()).ToArray()) },
         { (typeof(VBuffer<bool>), typeof(bool[])), new Func<object, object>(value => VBufferToArray<bool>(value)) },
         { (typeof(VBuffer<TimeSpan>), typeof(TimeSpan[])), new Func<object, object>(value => VBufferToArray<TimeSpan>(value)) },
         { (typeof(VBuffer<DateTime>), typeof(DateTime[])), new Func<object, object>(value => VBufferToArray<DateTime>(value)) },
         { (typeof(VBuffer<DateTimeOffset>), typeof(DateTimeOffset[])), new Func<object, object>(value => VBufferToArray<DateTimeOffset>(value)) },
         { (typeof(sbyte[]), typeof(VBuffer<sbyte>)), new Func<object, object>(value => ArrayToVBuffer(value as sbyte[])) },
         { (typeof(byte[]), typeof(VBuffer<byte>)), new Func<object, object>(value => ArrayToVBuffer(value as byte[])) },
         { (typeof(short[]), typeof(VBuffer<short>)), new Func<object, object>(value => ArrayToVBuffer(value as short[])) },
         { (typeof(ushort[]), typeof(VBuffer<ushort>)), new Func<object, object>(value => ArrayToVBuffer(value as ushort[])) },
         { (typeof(int[]), typeof(VBuffer<int>)), new Func<object, object>(value => ArrayToVBuffer(value as int[])) },
         { (typeof(uint[]), typeof(VBuffer<uint>)), new Func<object, object>(value => ArrayToVBuffer(value as uint[])) },
         { (typeof(long[]), typeof(VBuffer<long>)), new Func<object, object>(value => ArrayToVBuffer(value as long[])) },
         { (typeof(ulong[]), typeof(VBuffer<ulong>)), new Func<object, object>(value => ArrayToVBuffer(value as ulong[])) },
         { (typeof(float[]), typeof(VBuffer<float>)), new Func<object, object>(value => ArrayToVBuffer(value as float[])) },
         { (typeof(double[]), typeof(VBuffer<double>)), new Func<object, object>(value => ArrayToVBuffer(value as double[])) },
         { (typeof(string), typeof(ReadOnlyMemory<char>)), new Func<object, object>(value => new ReadOnlyMemory<char>(((string)value).ToCharArray())) },
         { (typeof(string[]), typeof(VBuffer<ReadOnlyMemory<char>>)), new Func<object, object>(value => { var values = ((string[])value).Select(s => new ReadOnlyMemory<char>(s.ToCharArray())).ToArray(); return new VBuffer<ReadOnlyMemory<char>>(values.Length, values); }) },
         { (typeof(bool[]), typeof(VBuffer<bool>)), new Func<object, object>(value => ArrayToVBuffer(value as bool[])) },
         { (typeof(TimeSpan[]), typeof(VBuffer<TimeSpan>)), new Func<object, object>(value => ArrayToVBuffer(value as TimeSpan[])) },
         { (typeof(DateTime[]), typeof(VBuffer<DateTime>)), new Func<object, object>(value => ArrayToVBuffer(value as DateTime[])) },
         { (typeof(DateTimeOffset[]), typeof(VBuffer<DateTimeOffset>)), new Func<object, object>(value => ArrayToVBuffer(value as DateTimeOffset[])) },
      };
      /// <summary>
      /// Valore
      /// </summary>
      public object Value { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="value">Valore</param>
      public DataValue(object value) => Value = value;
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="type">Tipo devinito nello schema delle viste di dati</param>
      /// <param name="value">Valore</param>
      public DataValue(DataViewType type, object value)
      {
         if (type.RawType == value.GetType())
            Value = value;
         else if (Converter.TryGetValue((type.RawType, value.GetType()), out var convertion))
            Value = convertion(value);
         else
            throw new InvalidCastException($"Cannot convert {value.GetType()} to {type.RawType}");
      }
      /// <summary>
      /// Converte da VBuffer ad array 
      /// </summary>
      /// <typeparam name="T">Tipo di VBuffer e di array</typeparam>
      /// <param name="array">L'array</param>
      /// <returns>Il VBuffer</returns>
      private static VBuffer<T> ArrayToVBuffer<T>(T[] array) => new VBuffer<T>(array.Length, array);
      /// <summary>
      /// Funzione di casting senza eccezione
      /// </summary>
      /// <typeparam name="T">Tipo a cui castare</typeparam>
      /// <returns>Il tipo castato o eccezione se non possibile</returns>
      public T As<T>() => CanConvert(Value.GetType(), typeof(T)) ? Convert<T>(Value) : default;
      /// <summary>
      /// Verifica se esiste una conversione possibile fra i valori
      /// </summary>
      /// <param name="from">Tipo sorgente</param>
      /// <param name="to">Tipo destinazione</param>
      /// <returns>true se conversione possibile</returns>
      public static bool CanConvert(Type from, Type to) => to.IsAssignableFrom(from) || Converter.ContainsKey((from, to));
      /// <summary>
      /// Converte un valore
      /// </summary>
      /// <typeparam name="T">Tipo in cui convertire</typeparam>
      /// <param name="value">Valore da convertire</param>
      /// <returns>Il valore convertito</returns>
      public static T Convert<T>(object value) => (T)Convert(value, typeof(T));
      /// <summary>
      /// Converte un valore
      /// </summary>
      /// <typeparam name="T">Tipo in cui convertire</typeparam>
      /// <param name="value">Valore da convertire</param>
      /// <param name="toType">Tipo in cui deve essere convertito</param>
      /// <returns>Il valore convertito</returns>
      public static object Convert(object value, Type toType)
      {
         if (Converter.TryGetValue((value.GetType(), toType), out Func<object, object> convertion))
            return convertion(value);
         if (toType.IsAssignableFrom(value.GetType()))
            return value;
         throw new InvalidCastException($"Type {value.GetType()} cannot be converted to {toType}");
      }
      /// <summary>
      /// Test di uguaglianza
      /// </summary>
      /// <param name="obj">Oggetto da testare</param>
      /// <returns>true se gli oggetti sono uguali</returns>
      public override bool Equals(object obj) => Value.Equals(obj);
      /// <summary>
      /// Restituisce il codice hash
      /// </summary>
      /// <returns>Il codice hash</returns>
      public override int GetHashCode() => Value.GetHashCode();
      /// <summary>
      /// Funzione di casting
      /// </summary>
      /// <typeparam name="T">Tipo a cui castare</typeparam>
      /// <returns>Il tipo castato o eccezione se non possibile</returns>
      public T To<T>() => Convert<T>(Value);
      /// <summary>
      /// Operatore di verifica uguaglianza
      /// </summary>
      /// <param name="left">Operando di sinistra</param>
      /// <param name="right">Operando di destra</param>
      /// <returns>true se uguali</returns>
      public static bool operator ==(DataValue left, DataValue right) => left.Equals(right);
      /// <summary>
      /// Operatore di verifica uguaglianza
      /// </summary>
      /// <param name="left">Operando di sinistra</param>
      /// <param name="right">Operando di destra</param>
      /// <returns>true se diversi</returns>
      public static bool operator !=(DataValue left, DataValue right) => !(left == right);
      /// <summary>
      /// Rappresentazione in formato stringa
      /// </summary>
      /// <returns></returns>
      public override string ToString() => Value.ToString();
      /// <summary>
      /// Converte da VBuffer ad array 
      /// </summary>
      /// <typeparam name="T">Tipo di VBuffer e di array</typeparam>
      /// <param name="vBuffer">Il buffer</param>
      /// <returns>L'array</returns>
      private static T[] VBufferToArray<T>(object vBuffer) => ((VBuffer<T>)vBuffer).DenseValues().ToArray();
      /// <summary>
      /// Operatore di conversione a sbyte
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator sbyte(DataValue value) => value.To<sbyte>();
      /// <summary>
      /// Operatore di conversione ad array di sbyte
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator sbyte[](DataValue value) => value.To<sbyte[]>();
      /// <summary>
      /// Operatore di conversione a byte
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator byte(DataValue value) => value.To<byte>();
      /// <summary>
      /// Operatore di conversione ad array di byte
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator byte[](DataValue value) => value.To<byte[]>();
      /// <summary>
      /// Operatore di conversione a short
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator short(DataValue value) => value.To<short>();
      /// <summary>
      /// Operatore di conversione ad array di short
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator short[](DataValue value) => value.To<short[]>();
      /// <summary>
      /// Operatore di conversione a ushort
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator ushort(DataValue value) => value.To<ushort>();
      /// <summary>
      /// Operatore di conversione ad array di ushort
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator ushort[](DataValue value) => value.To<ushort[]>();
      /// <summary>
      /// Operatore di conversione a int
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator int(DataValue value) => value.To<int>();
      /// <summary>
      /// Operatore di conversione ad array di int
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator int[](DataValue value) => value.To<int[]>();
      /// <summary>
      /// Operatore di conversione a uint
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator uint(DataValue value) => value.To<uint>();
      /// <summary>
      /// Operatore di conversione ad array di uint
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator uint[](DataValue value) => value.To<uint[]>();
      /// <summary>
      /// Operatore di conversione a long
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator long(DataValue value) => value.To<long>();
      /// <summary>
      /// Operatore di conversione ad array di long
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator long[](DataValue value) => value.To<long[]>();
      /// <summary>
      /// Operatore di conversione a ulong
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator ulong(DataValue value) => value.To<ulong>();
      /// <summary>
      /// Operatore di conversione ad array di ulong
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator ulong[](DataValue value) => value.To<ulong[]>();
      /// <summary>
      /// Operatore di conversione a float
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator float(DataValue value) => value.To<float>();
      /// <summary>
      /// Operatore di conversione ad array di float
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator float[](DataValue value) => value.To<float[]>();
      /// <summary>
      /// Operatore di conversione a double
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator double(DataValue value) => value.To<double>();
      /// <summary>
      /// Operatore di conversione ad array di double
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator double[](DataValue value) => value.To<double[]>();
      /// <summary>
      /// Operatore di conversione a stringa
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator string(DataValue value) => value.To<string>();
      /// <summary>
      /// Operatore di conversione ad array di stringhe
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator string[](DataValue value) => value.To<string[]>();
      /// <summary>
      /// Operatore di conversione a bool
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator bool(DataValue value) => value.To<bool>();
      /// <summary>
      /// Operatore di conversione ad array di bool
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator bool[](DataValue value) => value.To<bool[]>();
      /// <summary>
      /// Operatore di conversione a TimeSpan
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator TimeSpan(DataValue value) => value.To<TimeSpan>();
      /// <summary>
      /// Operatore di conversione ad array di TimeSpan
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator TimeSpan[](DataValue value) => value.To<TimeSpan[]>();
      /// <summary>
      /// Operatore di conversione a data
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator DateTime(DataValue value) => value.To<DateTime>();
      /// <summary>
      /// Operatore di conversione ad array di date
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator DateTime[](DataValue value) => value.To<DateTime[]>();
      /// <summary>
      /// Operatore di conversione a offset di data
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator DateTimeOffset(DataValue value) => value.To<DateTimeOffset>();
      /// <summary>
      /// Operatore di conversione ad array di offset di date
      /// </summary>
      /// <param name="value"></param>
      public static implicit operator DateTimeOffset[](DataValue value) => value.To<DateTimeOffset[]>();
      #endregion
   }
}
