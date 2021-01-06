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
      public DataValue(object value) => Value = value;
      /// <summary>
      /// Funzione di casting senza eccezione
      /// </summary>
      /// <typeparam name="T">Tipo a cui castare</typeparam>
      /// <returns>Il tipo castato o eccezione se non possibile</returns>
      public T As<T>()
      {
         if (typeof(T).IsAssignableFrom(Value.GetType()))
            return (T)Value;
         if (Converter.TryGetValue((Value.GetType(), typeof(T)), out Func<dynamic, dynamic> convertion))
            return convertion(Value);
         return default;
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
      public T To<T>()
      {
         if (typeof(T).IsAssignableFrom(Value.GetType()))
            return (T)Value;
         if (Converter.TryGetValue((Value.GetType(), typeof(T)), out Func<dynamic, dynamic> convertion))
            return convertion(Value);
         throw new InvalidCastException($"Type {Value.GetType().Name} cannot be converted to {typeof(T).Name}");
      }
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
