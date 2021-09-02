using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning.Data
{
   /// <summary>
   /// Costruttore di schemi di viste dati
   /// </summary>
   public class DataViewSchemaBuilder
   {
      #region Fields
      /// <summary>
      /// Dizionario di conversione da tipo standard a tipo vista di dati
      /// </summary>
      private static readonly Dictionary<Type, Func<DataViewType>> _typeToDataView = new()
      {
         { typeof(bool), () => BooleanDataViewType.Instance },
         { typeof(bool[]), () => new VectorDataViewType(BooleanDataViewType.Instance) },
         { typeof(sbyte), () => NumberDataViewType.SByte },
         { typeof(sbyte[]), () => new VectorDataViewType(NumberDataViewType.SByte) },
         { typeof(byte), () => NumberDataViewType.Byte },
         { typeof(byte[]), () => new VectorDataViewType(NumberDataViewType.Byte) },
         { typeof(short), () => NumberDataViewType.Int16 },
         { typeof(short[]), () => new VectorDataViewType(NumberDataViewType.Int16) },
         { typeof(ushort), () => NumberDataViewType.UInt16 },
         { typeof(ushort[]), () => new VectorDataViewType(NumberDataViewType.UInt16) },
         { typeof(int), () => NumberDataViewType.Int32 },
         { typeof(int[]), () => new VectorDataViewType(NumberDataViewType.Int32) },
         { typeof(uint), () => NumberDataViewType.UInt32 },
         { typeof(uint[]), () => new VectorDataViewType(NumberDataViewType.UInt32) },
         { typeof(long), () => NumberDataViewType.Int64 },
         { typeof(long[]), () => new VectorDataViewType(NumberDataViewType.Int64) },
         { typeof(ulong), () => NumberDataViewType.UInt64 },
         { typeof(ulong[]), () => new VectorDataViewType(NumberDataViewType.UInt64) },
         { typeof(float), () => NumberDataViewType.Single },
         { typeof(float[]), () => new VectorDataViewType(NumberDataViewType.Single) },
         { typeof(double), () => NumberDataViewType.Double },
         { typeof(double[]), () => new VectorDataViewType(NumberDataViewType.Double) },
         { typeof(string), () => TextDataViewType.Instance },
         { typeof(string[]), () => new VectorDataViewType(TextDataViewType.Instance) },
         { typeof(TimeSpan), () => TimeSpanDataViewType.Instance },
         { typeof(TimeSpan[]), () => new VectorDataViewType(TimeSpanDataViewType.Instance) },
         { typeof(DateTime), () => DateTimeDataViewType.Instance },
         { typeof(DateTime[]), () => new VectorDataViewType(DateTimeDataViewType.Instance) },
         { typeof(DateTimeOffset), () => DateTimeOffsetDataViewType.Instance },
         { typeof(DateTimeOffset[]), () => new VectorDataViewType(DateTimeOffsetDataViewType.Instance) },
      };
      /// <summary>
      /// Costruttore di schema
      /// </summary>
      private readonly DataViewSchema.Builder _builder = new();
      #endregion
      #region Methods
      /// <summary>
      /// Aggiunge un elemento
      /// </summary>
      /// <param name="item">Elemento</param>
      public void Add((string Name, object Value) item) => Add(new[] { item });
      /// <summary>
      /// Aggiunge un elemento
      /// </summary>
      /// <param name="item">Elemento</param>
      public void Add((string Name, Type Type) item) => Add(new[] { item });
      /// <summary>
      /// Aggiunge un set di elementi
      /// </summary>
      /// <param name="items">Elementi</param>
      public void Add(IEnumerable<(string Name, object Value)> items)
      {
         var columns = Build(items.ToArray());
         _builder.AddColumns(from c in columns select new DataViewSchema.DetachedColumn(c));
      }
      /// <summary>
      /// Aggiunge un set di elementi
      /// </summary>
      /// <param name="items">Elementi</param>
      public void Add(IEnumerable<(string Name, Type Type)> items)
      {
         var columns = Build(items.ToArray());
         _builder.AddColumns(from c in columns select new DataViewSchema.DetachedColumn(c));
      }
      /// <summary>
      /// Costruisce lo schema a partire dai tipi
      /// </summary>
      /// <param name="items">Elementi</param>
      /// <returns>Lo schema</returns>
      public static DataViewSchema Build(params (string Name, Type Type)[] items)
      {
         var builder = new DataViewSchema.Builder();
         foreach (var item in items) {
            if (string.IsNullOrWhiteSpace(item.Name))
               throw new ArgumentException("The name of the columns cannot be undefined", nameof(items));
            if (!_typeToDataView.TryGetValue(item.Type, out var converter))
               throw new ArgumentException($"The type {item.Type} of the column {item.Name} is not convertible to a DataViewType", nameof(items));
            builder.AddColumn(item.Name, converter());
         }
         return builder.ToSchema();
      }
      /// <summary>
      /// Costruisce lo schema a partire dai tipi
      /// </summary>
      /// <param name="items">Elementi</param>
      /// <returns>Lo schema</returns>
      public static DataViewSchema Build(params (string Name, object Value)[] items)
      {
         var list = new List<(string Name, Type Type)>();
         foreach (var (Name, Value) in items) {
            if (Value == null)
               throw new ArgumentException($"The value of the column {Name} cannot be null", nameof(items));
            list.Add((Name, Value.GetType()));
         }
         return Build(list.ToArray());
      }
      #endregion
   }
}
