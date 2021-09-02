using MachineLearning.Data;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni alla DataViewSchema
   /// </summary>
   public static class DataViewSchemaExtensions
   {
      private static readonly Dictionary<DataViewType, DataKind> _toDataKind = new Dictionary<DataViewType, DataKind>
      {
         { BooleanDataViewType.Instance, DataKind.Boolean },
         { NumberDataViewType.SByte, DataKind.SByte },
         { NumberDataViewType.Byte, DataKind.Byte },
         { NumberDataViewType.Int16, DataKind.Int16 },
         { NumberDataViewType.UInt16, DataKind.UInt16 },
         { NumberDataViewType.Int32, DataKind.Int32 },
         { NumberDataViewType.UInt32, DataKind.UInt32 },
         { NumberDataViewType.Int64, DataKind.Int64 },
         { NumberDataViewType.UInt64, DataKind.UInt64 },
         { NumberDataViewType.Single, DataKind.Single },
         { NumberDataViewType.Double, DataKind.Double },
         { TextDataViewType.Instance, DataKind.String },
         { DateTimeDataViewType.Instance, DataKind.DateTime },
         { DateTimeOffsetDataViewType.Instance, DataKind.DateTimeOffset },
         { TimeSpanDataViewType.Instance, DataKind.TimeSpan },
      };
      #region Methods
      /// <summary>
      /// Crea un elenco di colonne di caricamento dati testuali a partire da uno schema di vista
      /// </summary>
      /// <param name="schema">Schema della vista dati</param>
      /// <returns>Le colonne di caricamento testi</returns>
      public static TextLoader.Column[] ToTextLoaderColumns(this DataViewSchema schema)
      {
         var columns = new TextLoader.Column[schema.Count];
         for (var i = 0; i < columns.Length; i++) {
            if (!_toDataKind.TryGetValue(schema[i].Type, out var textLoaderType))
               throw new ArgumentException($"The type {schema[i].Type} cannot be converted in a {nameof(DataKind)} type in schema's column {schema[i].Name}[{schema[i].Index}]", nameof(schema));
            columns[i] = new TextLoader.Column(schema[i].Name, textLoaderType, schema[i].Index);
         }
         return columns;
      }
      /// <summary>
      /// Crea un elenco di colonne di caricamento dati testuali a partire da uno schema di vista
      /// </summary>
      /// <param name="schema">Schema della vista dati</param>
      /// <returns>Le colonne di caricamento testi</returns>
      public static TextLoader.Column[] ToTextLoaderColumns(this DataSchema schema) => ToTextLoaderColumns((DataViewSchema)schema);
      #endregion
   }
}
